"""Metrik ve bölme testleri (elle hesaplanmış örnekler)."""
import numpy as np
import pytest

from src.data_split import split_per_student
from src.metrics import evaluate


def test_metrikler_elle_hesap():
    R = np.array([[1, 1, 0, 0]])
    P = np.array([[0.9, 0.2, 0.7, 0.1]])      # tahmin: 1, 0, 1, 0 -> TP=1 FP=1 FN=1
    m = evaluate(R, P, np.ones_like(R, bool))
    assert m["PRE"] == pytest.approx(0.5)
    assert m["Recall"] == pytest.approx(0.5)
    assert m["F1"] == pytest.approx(0.5)
    assert m["MAE"] == pytest.approx((0.1 + 0.8 + 0.7 + 0.1) / 4)
    assert m["RMSE"] == pytest.approx(np.sqrt((0.01 + 0.64 + 0.49 + 0.01) / 4))


def test_hic_pozitif_tahmin_yoksa_pre_sifir():
    R = np.array([[1, 0]])
    m = evaluate(R, np.array([[0.1, 0.2]]), np.ones_like(R, bool))
    assert m["PRE"] == 0 and m["Recall"] == 0 and m["F1"] == 0


def test_mask_sadece_test_hucrelerini_kullanir():
    R = np.array([[1, 0]])
    P = np.array([[0.9, 0.9]])
    m = evaluate(R, P, np.array([[True, False]]))
    assert m["MAE"] == pytest.approx(0.1)


def test_bolme_oran_ve_tekrarlanabilirlik():
    a = split_per_student((100, 20), 0.6, seed=3)
    assert (a.sum(1) == 12).all()
    assert np.array_equal(a, split_per_student((100, 20), 0.6, seed=3))
    assert not np.array_equal(a, split_per_student((100, 20), 0.6, seed=4))


def test_bolme_her_ogrencide_egitim_ve_test_var():
    a = split_per_student((10, 20), 0.9, seed=0)
    assert (a.sum(1) >= 1).all() and ((~a).sum(1) >= 1).all()


def test_auc_elle_hesap():
    from src.metrics import auc
    assert auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)
    assert auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)
    # 4 (pozitif, negatif) çiftinden 3'ü doğru sıralı -> 0.75
    assert auc([1, 1, 0, 0], [0.9, 0.3, 0.5, 0.1]) == pytest.approx(0.75)
    assert auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)          # eşit skor = şans
    assert auc([1, 1], [0.2, 0.9]) == 0.5                          # tek sınıf


def test_kalibrasyon_alanlari():
    R = np.array([[1, 0, 1, 0]])
    m = evaluate(R, np.array([[0.9, 0.8, 0.7, 0.6]]), np.ones_like(R, bool))
    assert m["ort_P"] == pytest.approx(0.75) and m["ort_R"] == pytest.approx(0.5)
