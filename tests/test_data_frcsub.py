"""FrcSub doğrulama ve yükleme testleri."""
import numpy as np
import pytest

from src.data_frcsub import (FRCSUB_DIR, N_ITEMS, N_SKILLS, N_STUDENTS,
                             load_frcsub, validate_frcsub)


def _gecerli():
    R = np.zeros((N_STUDENTS, N_ITEMS), dtype=int)
    Q = np.ones((N_ITEMS, N_SKILLS), dtype=int)
    return R, Q


def test_gecerli_veri_hata_vermez():
    validate_frcsub(*_gecerli())


def test_yanlis_boyut_hata_verir():
    R, Q = _gecerli()
    with pytest.raises(ValueError):
        validate_frcsub(R[:-1], Q)


def test_ikili_olmayan_cevap_hata_verir():
    R, Q = _gecerli()
    R[0, 0] = 2
    with pytest.raises(ValueError):
        validate_frcsub(R, Q)


def test_becerisiz_soru_hata_verir():
    R, Q = _gecerli()
    Q[3] = 0
    with pytest.raises(ValueError):
        validate_frcsub(R, Q)


@pytest.mark.skipif(not FRCSUB_DIR.exists(),
                    reason="FrcSub indirilmemiş: python -m src.data_frcsub")
def test_gercek_frcsub_yuklenir():
    R, Q = load_frcsub()
    assert R.shape == (536, 20)
    assert Q.shape == (20, 8)
