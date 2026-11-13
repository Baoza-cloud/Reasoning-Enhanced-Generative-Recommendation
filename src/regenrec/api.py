from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from regenrec.data import Interaction, iter_sequence_examples_jsonl
from regenrec.serving import RecommendationEngine, ServingSettings


class HistoryEventRequest(BaseModel):
    item_id: Annotated[str, Field(min_length=1, max_length=64)]
    behavior: Annotated[int, Field(ge=0)] = 3
    behavior_intensity: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    time_gap_bucket: Annotated[int, Field(ge=0)] = 0

    def to_domain(self) -> Interaction:
        return Interaction(
            item_id=self.item_id,
            behavior=self.behavior,
            behavior_intensity=self.behavior_intensity,
            time_gap_bucket=self.time_gap_bucket,
        )


class RecommendationRequest(BaseModel):
    history: Annotated[list[HistoryEventRequest], Field(min_length=1)]
    beam_size: Annotated[int, Field(ge=1)] = 20
    long_tail_mode: Literal["off", "mild", "aggressive"] = "off"


class BatchHistoryRequest(BaseModel):
    request_id: Annotated[str, Field(min_length=1, max_length=128)]
    history: Annotated[list[HistoryEventRequest], Field(min_length=1)]


class BatchRecommendationRequest(BaseModel):
    requests: Annotated[list[BatchHistoryRequest], Field(min_length=1)]
    beam_size: Annotated[int, Field(ge=1)] = 20
    long_tail_mode: Literal["off", "mild", "aggressive"] = "off"


def _engine(request: Request) -> RecommendationEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="recommendation model is not ready")
    return engine


def _as_http_error(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


def create_app(
    *,
    settings: ServingSettings | None = None,
    engine: RecommendationEngine | None = None,
) -> FastAPI:
    resolved_settings = settings or ServingSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is not None:
            app.state.engine = engine
        else:
            app.state.engine = await run_in_threadpool(
                RecommendationEngine.load, resolved_settings
            )
        yield
        app.state.engine = None

    app = FastAPI(
        title="ReGenRec Online Inference",
        version="1.0.0",
        description="Catalog-constrained Top-10 product recommendation service.",
        lifespan=lifespan,
    )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(request: Request) -> dict[str, object]:
        active = _engine(request)
        return {
            "status": "ready",
            "catalog_items": len(active.catalog.item_ids),
            "device": str(active.device),
            "max_beam_size": active.settings.max_beam_size,
            "long_tail_modes": ["off", "mild", "aggressive"],
        }

    @app.post("/v1/recommend")
    async def recommend(
        payload: RecommendationRequest, request: Request
    ) -> dict[str, object]:
        active = _engine(request)
        try:
            rows = await run_in_threadpool(
                active.recommend,
                [[event.to_domain() for event in payload.history]],
                beam_size=payload.beam_size,
                long_tail_mode=payload.long_tail_mode,
            )
        except ValueError as exc:
            raise _as_http_error(exc) from exc
        return rows[0]

    @app.post("/v1/recommend/batch")
    async def recommend_batch(
        payload: BatchRecommendationRequest, request: Request
    ) -> dict[str, object]:
        active = _engine(request)
        histories = [
            [event.to_domain() for event in item.history] for item in payload.requests
        ]
        try:
            rows = await run_in_threadpool(
                active.recommend,
                histories,
                beam_size=payload.beam_size,
                long_tail_mode=payload.long_tail_mode,
            )
        except ValueError as exc:
            raise _as_http_error(exc) from exc
        return {
            "results": [
                {"request_id": item.request_id, **row}
                for item, row in zip(payload.requests, rows, strict=True)
            ]
        }

    @app.get("/v1/example")
    async def example(request: Request) -> dict[str, object]:
        active = _engine(request)
        test_path = active.config["data"]["test_path"]
        sequence = next(iter_sequence_examples_jsonl(test_path))
        return {
            "history": [
                {
                    "item_id": event.item_id,
                    "title": active.item_metadata[event.item_id]["title"],
                    "behavior": event.behavior,
                    "behavior_intensity": event.behavior_intensity,
                    "time_gap_bucket": event.time_gap_bucket,
                }
                for event in sequence.history
            ],
            "source": "Amazon Reviews 2023 test split",
            "target_hidden": True,
        }

    @app.get("/demo", response_class=HTMLResponse)
    async def demo() -> str:
        return DEMO_HTML

    return app


DEMO_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ReGenRec 推理演示</title>
  <style>
    body{font-family:system-ui,sans-serif;max-width:1180px;margin:28px auto;padding:0 18px;background:#f6f7fb;color:#172033}
    h1{margin-bottom:5px}.hint{color:#5d6678}.bar{display:flex;gap:12px;align-items:center;margin:18px 0}
    button,select{padding:9px 13px;border-radius:8px;border:1px solid #bdc5d4;background:white}button{background:#2457d6;color:white;cursor:pointer}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:white;border-radius:12px;padding:16px;box-shadow:0 3px 14px #1c29451a;overflow:auto}
    pre{white-space:pre-wrap;word-break:break-word;font-size:12px;max-height:360px;overflow:auto;margin-bottom:0}.wide{grid-column:1/-1}.error{color:#b42318}
    @media(max-width:800px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}}
  </style>
</head>
<body>
  <h1>生成式推荐在线推理</h1>
  <div class="hint">真实测试序列仅作为输入示例，目标商品不会返回给页面。</div>
  <div class="bar">
    <label>Beam <select id="beam"><option>1</option><option>10</option><option selected>20</option></select></label>
    <label>长尾模式 <select id="tail"><option value="off">关闭</option><option value="mild">温和</option><option value="aggressive">增强</option></select></label>
    <button id="run">运行推理</button><span id="status"></span>
  </div>
  <div class="grid">
    <section class="card"><h3>用户行为序列</h3><pre id="history">载入中…</pre></section>
    <section class="card"><h3>长期 / 近期偏好</h3><pre id="preference">等待运行</pre></section>
    <section class="card"><h3>Semantic ID 与 Beam 候选</h3><pre id="beamOutput">等待运行</pre></section>
    <section class="card"><h3>最终 Top-10 商品</h3><pre id="recommendations">等待运行</pre></section>
  </div>
<script>
let historyEvents=[];
const show=(id,value)=>document.getElementById(id).textContent=JSON.stringify(value,null,2);
async function loadExample(){
  const response=await fetch('/v1/example'); const data=await response.json();
  historyEvents=data.history.map(({title,...event})=>event); show('history',data.history);
}
async function run(){
  const status=document.getElementById('status'); status.textContent='推理中…'; status.className='';
  try{
    const response=await fetch('/v1/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({history:historyEvents,beam_size:Number(document.getElementById('beam').value),long_tail_mode:document.getElementById('tail').value})});
    const data=await response.json(); if(!response.ok)throw new Error(JSON.stringify(data));
    show('preference',data.preference_basis); show('beamOutput',data.beam_candidates); show('recommendations',data.recommendations);
    status.textContent=`完成：${data.latency_ms.toFixed(1)} ms，返回 ${data.returned_count} 个商品`;
  }catch(error){status.textContent=error.message;status.className='error'}
}
document.getElementById('run').addEventListener('click',run); loadExample().catch(error=>document.getElementById('status').textContent=error.message);
</script>
</body></html>
"""


app = create_app()
