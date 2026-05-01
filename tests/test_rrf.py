from hierarchical_rag.retrieval import rrf_fuse


def test_rrf_formula_single_ranking():
    out = rrf_fuse([[10, 20, 30]], k=60)
    # 1-indexed rank: rank 1 → 1/61, rank 2 → 1/62, rank 3 → 1/63
    assert abs(out[10] - 1 / 61) < 1e-9
    assert abs(out[20] - 1 / 62) < 1e-9
    assert abs(out[30] - 1 / 63) < 1e-9


def test_rrf_combines_two_rankings():
    out = rrf_fuse([[1, 2, 3], [3, 2, 1]], k=60)
    # id 1: 1/61 + 1/63;  id 3: 1/63 + 1/61 → equal to id 1
    # id 2: 1/62 + 1/62 → smaller than 1/61 + 1/63 (cross-rank confirmation wins)
    assert abs(out[1] - out[3]) < 1e-9
    assert out[1] > out[2]


def test_rrf_empty_input():
    assert rrf_fuse([], k=60) == {}
    assert rrf_fuse([[]], k=60) == {}


def test_rrf_k_changes_magnitude_not_order():
    a = rrf_fuse([[1, 2, 3, 4, 5]], k=10)
    b = rrf_fuse([[1, 2, 3, 4, 5]], k=60)
    # ordering invariant
    assert sorted(a.items(), key=lambda kv: -kv[1]) == \
           [(1, a[1]), (2, a[2]), (3, a[3]), (4, a[4]), (5, a[5])]
    assert sorted(b.items(), key=lambda kv: -kv[1]) == \
           [(1, b[1]), (2, b[2]), (3, b[3]), (4, b[4]), (5, b[5])]
    # smaller k → larger absolute scores
    assert a[1] > b[1]
