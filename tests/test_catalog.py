import torch

from regenrec.catalog import SemanticIdCatalog


def test_catalog_disambiguates_collisions_and_constrains_prefixes() -> None:
    item_ids = ["a", "b", "c"]
    codes = torch.tensor([[1, 2], [1, 2], [0, 3]])
    catalog = SemanticIdCatalog.from_codes(item_ids, codes, codebook_size=4)

    a = catalog.tokens_for_item("a")
    b = catalog.tokens_for_item("b")
    assert not torch.equal(a, b)
    assert torch.equal(a[:-1], b[:-1])
    assert catalog.resolve(a.tolist()) == "a"
    assert catalog.resolve([catalog.special_tokens.bos, *b.tolist(), catalog.special_tokens.eos]) == "b"
    assert set(catalog.allowed_next([catalog.special_tokens.bos, *a[:-1].tolist()])) == {
        int(a[-1]),
        int(b[-1]),
    }
    assert catalog.allowed_next([catalog.special_tokens.bos, *a.tolist()]) == [
        catalog.special_tokens.eos
    ]


def test_catalog_round_trip(tmp_path) -> None:
    catalog = SemanticIdCatalog.from_codes(
        ["a", "b"], torch.tensor([[0, 1], [1, 0]]), codebook_size=2
    )
    path = tmp_path / "catalog.json"
    catalog.save(path)
    restored = SemanticIdCatalog.load(path)
    assert restored.to_dict() == catalog.to_dict()
