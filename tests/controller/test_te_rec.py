import numpy as np

import cohere_core.controller.phasing as phasing


class FakeComm:
    def __init__(self, size=3, rank=1, recv_values=None):
        self._size = size
        self._rank = rank
        self.sent = []
        self.recv_values = list(recv_values or [])

    def Get_size(self):
        return self._size

    def Get_rank(self):
        return self._rank

    def send(self, value, dest):
        self.sent.append((dest, value))

    def recv(self, source):
        return self.recv_values.pop(0)


def te_params():
    return {
        "algorithm_sequence": "ER",
        "weight": 0.25,
        "hio_beta": 0.5,
    }


def test_te_rec_init_sets_comm_fields(fake_devlib):
    comm = FakeComm(size=4, rank=2)

    rec = phasing.TeRec(te_params(), "data.npy", "np", comm)

    assert rec.comm is comm
    assert rec.size == 4
    assert rec.rank == 2
    assert rec.weight == 0.25


def test_exchange_data_info_for_full_data_middle_rank(fake_devlib):
    comm = FakeComm(size=3, rank=1, recv_values=[False, True])
    rec = phasing.TeRec(te_params(), "data.npy", "np", comm)
    rec.data = np.ones((2, 2, 2))

    rec.exchange_data_info()
    assert rec.is_full_data is True
    assert rec.send_to_prev is False
    assert rec.send_to_next is True
    assert len(comm.sent) == 2


def test_exchange_data_info_for_partial_data_middle_rank(fake_devlib):
    comm = FakeComm(size=3, rank=1, recv_values=[True, False])
    rec = phasing.TeRec(te_params(), "data.npy", "np", comm)
    rec.data = np.array([[[1, -1]]])

    rec.exchange_data_info()

    assert rec.is_full_data is False
    assert rec.send_to_prev is True
    assert rec.send_to_next is False


def test_te_rec_er_full_data_behaves_like_base_er(fake_devlib):
    comm = FakeComm(size=3, rank=1)
    rec = phasing.TeRec(te_params(), "data.npy", "np", comm)
    rec.is_full_data = True
    rec.send_to_prev = False
    rec.send_to_next = False
    rec.ds_image_proj = np.array([[1.0, 2.0]])
    rec.support = np.array([[1.0, 0.0]])

    rec.er()

    np.testing.assert_allclose(rec.ds_image, np.array([[1.0, 0.0]]))


def test_te_rec_er_partial_data_uses_neighbors(fake_devlib):
    comm = FakeComm(size=3, rank=1, recv_values=[np.array([[4.0, 4.0]]), np.array([[2.0, 2.0]])])
    rec = phasing.TeRec(te_params(), "data.npy", "np", comm)
    rec.is_full_data = False
    rec.send_to_prev = True
    rec.send_to_next = True
    rec.ds_image = np.array([[10.0, 10.0]])
    rec.ds_image_proj = np.array([[1.0, 2.0]])
    rec.support = np.array([[1.0, 1.0]])

    rec.er()

    expected = (1 / (1 + 2 * 0.25)) * rec.support * (np.array([[1.0, 2.0]]) + 0.25 * (np.array([[2.0, 2.0]]) + np.array([[4.0, 4.0]])))
    np.testing.assert_allclose(rec.ds_image, expected)
    assert len(comm.sent) == 2


def test_te_rec_hio_full_data(fake_devlib):
    comm = FakeComm(size=3, rank=1)
    rec = phasing.TeRec(te_params(), "data.npy", "np", comm)
    rec.is_full_data = True
    rec.send_to_prev = False
    rec.send_to_next = False
    rec.ds_image = np.array([[10.0, 20.0]])
    rec.ds_image_proj = np.array([[1.0, 2.0]])
    rec.support = np.array([[1.0, 0.0]])

    rec.hio()

    np.testing.assert_allclose(rec.ds_image, np.array([[1.0, 19.0]]))


def test_te_rec_hio_partial_data(fake_devlib):
    comm = FakeComm(size=3, rank=1, recv_values=[np.array([[6.0, 6.0]]), np.array([[2.0, 2.0]])])
    rec = phasing.TeRec(te_params(), "data.npy", "np", comm)
    rec.is_full_data = False
    rec.send_to_prev = True
    rec.send_to_next = True
    rec.ds_image = np.array([[10.0, 20.0]])
    rec.ds_image_proj = np.array([[1.0, 2.0]])
    rec.support = np.array([[1.0, 0.0]])

    rec.hio()

    base = np.array([[1.0, 19.0]])
    corr = 0.25 * rec.support * (2 * np.array([[10.0, 20.0]]) - (np.array([[2.0, 2.0]]) + np.array([[6.0, 6.0]])))
    expected = base - corr
    np.testing.assert_allclose(rec.ds_image, expected)
    assert len(comm.sent) == 2


def test_te_rec_modulus_does_not_modify_missing_frames(fake_devlib, monkeypatch):
    comm = FakeComm()
    rec = phasing.TeRec(te_params(), "data.npy", "np", comm)
    rec.iter_data = np.array([[2.0, -1.0]])
    rec.rs_amplitudes = np.array([[1.0, 3.0]])
    rec.errs = []

    monkeypatch.setattr(phasing.dvut, "get_norm", lambda arr: np.linalg.norm(arr))

    rec.modulus()

    assert len(rec.errs) == 1
    assert rec.rs_amplitudes[0, 0] == 2.0
    assert rec.rs_amplitudes[0, 1] == 3.0