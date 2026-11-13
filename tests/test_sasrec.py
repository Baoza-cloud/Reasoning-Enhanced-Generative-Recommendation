import torch

from regenrec.models.sasrec import SASRec


def test_sasrec_forward_backward_and_recommendation() -> None:
    torch.manual_seed(17)
    model = SASRec(
        num_items=6,
        model_dim=16,
        num_heads=4,
        num_layers=1,
        feedforward_dim=32,
        dropout=0.0,
        max_history_length=4,
    )
    history = torch.tensor([[1, 2, 0], [2, 3, 4]])
    mask = history.ne(0)
    targets = torch.tensor([2, 4])
    logits = model(history, mask)
    assert logits.shape == (2, 6)
    loss = model.loss(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    scores, indices = model.recommend(history, mask, k=3)
    assert scores.shape == indices.shape == (2, 3)
    assert torch.all(indices >= 0)
    assert torch.all(indices < 6)
