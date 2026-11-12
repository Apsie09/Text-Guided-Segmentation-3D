from src.data.refcoco_m import DEFAULT_SPLIT_FILE, load_split, make_group_split


def test_split_is_disjoint_complete_and_deterministic():
    ids = list(range(1000)) + list(range(500))  # duplicates must collapse
    a = make_group_split(ids, seed=0)
    b = make_group_split(ids, seed=0)
    assert a == b
    tr, va, te = (set(a[k]) for k in ("train", "val", "test"))
    assert not (tr & va or tr & te or va & te)
    assert tr | va | te == set(range(1000))
    assert (len(tr), len(va), len(te)) == (700, 150, 150)


def test_different_seed_gives_different_split():
    ids = range(1000)
    assert make_group_split(ids, seed=0)["test"] != make_group_split(ids, seed=1)["test"]


def test_saved_split_file_is_disjoint():
    s = load_split(DEFAULT_SPLIT_FILE)
    tr, va, te = (set(s[k]) for k in ("train", "val", "test"))
    assert not (tr & va or tr & te or va & te)
