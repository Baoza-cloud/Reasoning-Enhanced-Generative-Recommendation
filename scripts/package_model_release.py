from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

from regenrec.catalog import SemanticIdCatalog
from regenrec.serving import ServingSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the frozen generator and Semantic-ID catalog for release"
    )
    parser.add_argument("--version", default="v2.4-seed2026")
    parser.add_argument("--output-dir", type=Path, default=Path("release_assets"))
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("release/model_manifest.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_row(path: Path) -> dict[str, object]:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    settings = ServingSettings.from_env()
    settings.validate(require_long_tail=False)
    generator_path = settings.artifact_dir / "generator.pt"
    catalog_path = settings.artifact_dir / "catalog.json"
    catalog = SemanticIdCatalog.load(catalog_path)
    manifest = {
        "schema_version": 1,
        "release": args.version,
        "model": "token_loss_010",
        "seed": 2026,
        "dataset": "Amazon Reviews 2023 / Musical Instruments",
        "catalog_items": len(catalog.item_ids),
        "semantic_id": {
            "rq_levels": catalog.num_levels,
            "codebook_size": catalog.codebook_size,
            "sequence_length": catalog.sequence_length,
        },
        "files": [file_row(generator_path), file_row(catalog_path)],
        "required_config": "configs/v2_final_default.yaml",
        "long_tail_assets_included": False,
        "notes": (
            "BLaIR item embeddings and Amazon data are not redistributed; "
            "generate them with docs/DATA_AND_MODEL_SETUP.md."
        ),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / f"regenrec-{args.version}-model.tar.gz"
    with tempfile.TemporaryDirectory(prefix="regenrec-release-") as temporary:
        bundle = Path(temporary) / f"regenrec-{args.version}-model"
        bundle.mkdir()
        shutil.copy2(generator_path, bundle / "generator.pt")
        shutil.copy2(catalog_path, bundle / "catalog.json")
        shutil.copy2(args.manifest_output, bundle / "model_manifest.json")
        shutil.copy2(settings.config_path, bundle / "config.yaml")
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(bundle, arcname=bundle.name)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "sha256": sha256(archive_path),
                "checksum": str(checksum_path),
                "manifest": str(args.manifest_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
