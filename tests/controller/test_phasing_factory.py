import cohere_core.controller.phasing as phasing


def test_set_lib_from_pkg_sets_all_dependencies(monkeypatch):
    fake_lib = object()
    calls = {}

    monkeypatch.setattr(phasing.ut, "get_lib", lambda pkg: fake_lib)
    monkeypatch.setattr(phasing.ft, "set_lib", lambda lib: calls.setdefault("ft", lib))
    monkeypatch.setattr(phasing.dvut, "set_lib_from_pkg", lambda pkg: calls.setdefault("dvut", pkg))

    phasing.set_lib_from_pkg("np")

    assert phasing.devlib is fake_lib
    assert calls["ft"] is fake_lib
    assert calls["dvut"] == "np"


def test_create_rec_returns_basic_worker_on_success(monkeypatch, fake_devlib):
    monkeypatch.setattr(phasing.Rec, "init_dev", lambda self, dev: 0)
    monkeypatch.setattr(phasing.Rec, "init_iter_loop", lambda self, continue_dir=None: 0)

    worker = phasing.create_rec({"algorithm_sequence": "ER"}, "data.npy", "np", -1)

    assert worker is not None
    assert isinstance(worker, phasing.Rec)


def test_create_rec_returns_coupled_worker_when_rec_type_mp(monkeypatch, fake_devlib):
    monkeypatch.setattr(phasing.CoupledRec, "init_dev", lambda self, dev: 0)
    monkeypatch.setattr(phasing.CoupledRec, "init_iter_loop", lambda self, continue_dir=None: 0)

    worker = phasing.create_rec(
        {"algorithm_sequence": "ER"},
        ["peak1", "peak2"],
        "np",
        -1,
        rec_type="mp",
    )

    assert worker is not None
    assert isinstance(worker, phasing.CoupledRec)


def test_create_rec_returns_none_when_init_dev_fails(monkeypatch, fake_devlib):
    monkeypatch.setattr(phasing.Rec, "init_dev", lambda self, dev: -1)

    worker = phasing.create_rec({"algorithm_sequence": "ER"}, "data.npy", "np", -1)

    assert worker is None


def test_create_rec_returns_none_when_init_iter_loop_fails(monkeypatch, fake_devlib):
    monkeypatch.setattr(phasing.Rec, "init_dev", lambda self, dev: 0)
    monkeypatch.setattr(phasing.Rec, "init_iter_loop", lambda self, continue_dir=None: -1)

    worker = phasing.create_rec({"algorithm_sequence": "ER"}, "data.npy", "np", -1)

    assert worker is None


def test_create_rec_passes_continue_dir_to_init_iter_loop(monkeypatch, fake_devlib):
    captured = {}

    monkeypatch.setattr(phasing.Rec, "init_dev", lambda self, dev: 0)

    def fake_init_iter_loop(self, continue_dir=None):
        captured["continue_dir"] = continue_dir
        return 0

    monkeypatch.setattr(phasing.Rec, "init_iter_loop", fake_init_iter_loop)

    worker = phasing.create_rec(
        {"algorithm_sequence": "ER", "continue_dir": "prev_dir"},
        "data.npy",
        "np",
        -1,
    )

    assert worker is not None
    assert captured["continue_dir"] == "prev_dir"