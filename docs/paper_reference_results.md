# Makale referans sonuçları (Zhang & Wang, Sci. Rep. 15:11531, 2025)

Bu dosya, raporda yapılacak **her makale karşılaştırmasının tek kaynağıdır**. Değerler makalenin Tablo 4'ünden
elle aktarılmıştır (PDF repoda yoktur). Rapora girecek makale sayıları yalnızca buradan alınır.
Makalenin kendi tutarsızlıkları "Notlar" bölümündedir. Bizim sayılarımız bu dosyada değil, `results/tables/` altındadır.

## Tablo 4 (makaleden aktarım)

Sütunlar: Nesnel sorular %50 / %60 / %70 | Öznel sorular %50 / %60 / %70 (eğitim oranı).

| Ölçüt | Model | Nesnel 50 | Nesnel 60 | Nesnel 70 | Öznel 50 | Öznel 60 | Öznel 70 |
|---|---|---|---|---|---|---|---|
| PRE | DINA | 0.601 | 0.683 | 0.706 | 0.514 | 0.543 | 0.573 |
| PRE | FuzzyCDF | 0.654 | 0.724 | 0.736 | 0.571 | 0.584 | 0.594 |
| PRE | My-FuzzyCDF | 0.673 | 0.746 | 0.761 | 0.622 | 0.647 | 0.652 |
| PRE | DKT | 0.753 | 0.767 | 0.793 | 0.668 | 0.682 | 0.693 |
| PRE | C&RM-MAKT | 0.792 | 0.830 | 0.847 | 0.721 | 0.724 | 0.769 |
| PRE | MFCD-MAKT | 0.883 | 0.894 | 0.901 | 0.782 | 0.790 | 0.804 |
| Recall | DINA | 0.334 | 0.387 | 0.372 | 0.306 | 0.329 | 0.407 |
| Recall | FuzzyCDF | 0.387 | 0.462 | 0.442 | 0.335 | 0.364 | 0.424 |
| Recall | My-FuzzyCDF | 0.413 | 0.468 | 0.462 | 0.442 | 0.482 | 0.491 |
| Recall | DKT | 0.525 | 0.552 | 0.583 | 0.484 | 0.490 | 0.503 |
| Recall | C&RM-MAKT | 0.553 | 0.602 | 0.632 | 0.532 | 0.552 | 0.573 |
| Recall | MFCD-MAKT | 0.585 | 0.631 | 0.644 | 0.525 | 0.572 | 0.586 |
| F1 | DINA | 0.442 | 0.461 | 0.479 | 0.321 | 0.388 | 0.418 |
| F1 | FuzzyCDF | 0.483 | 0.557 | 0.537 | 0.388 | 0.463 | 0.516 |
| F1 | My-FuzzyCDF | 0.552 | 0.573 | 0.561 | 0.587 | 0.603 | 0.624 |
| F1 | DKT | 0.643 | 0.672 | 0.704 | 0.593 | 0.606 | 0.618 |
| F1 | C&RM-MAKT | 0.668 | 0.673 | 0.702 | 0.622 | 0.632 | 0.657 |
| F1 | MFCD-MAKT | 0.701 | 0.726 | 0.773 | 0.664 | 0.712 | 0.733 |

(Makalede son satır "MFCD-MAKTM" olarak yazılmıştır.)

## Türetilmiş fark: My-FuzzyCDF − FuzzyCDF (zaman + süre eklemenin makaledeki payı)

| Ölçüt | Nesnel 50 | Nesnel 60 | Nesnel 70 | Öznel 50 | Öznel 60 | Öznel 70 |
|---|---|---|---|---|---|---|
| PRE | +0.019 | +0.022 | +0.025 | +0.051 | +0.063 | +0.058 |
| Recall | +0.026 | +0.006 | +0.020 | +0.107 | +0.118 | +0.067 |
| F1 | +0.069 | +0.016 | +0.024 | +0.199 | +0.140 | +0.108 |

**Okuma:** Nesnel sorularda zaman/süre eklemenin makaledeki payı küçük, öznel sorularda belirgin (özellikle F1 ve
Recall). Önceki "fark küçük" ifademiz yalnızca nesnel sorular için doğruydu; düzeltildi (`docs/assumptions.md`).
Bizim FrcSub deneyimiz yalnızca nesnel sorular içerdiğinden zaman/süre etkisini zaten ölçemez.

## Notlar (makalenin kendi tutarsızlıkları)

- Metin, MFCD-MAKT'ın "her tipte ve oranda en iyi" Recall'a sahip olduğunu söyler; oysa öznel %50 Recall'da
  C&RM-MAKT (0.532) MFCD-MAKT'tan (0.525) yüksektir.
- PRE ve Recall'dan hesaplanan F1 tablodaki F1 ile bire bir tutmaz (ör. MFCD-MAKT nesnel %60: 0.894 ve 0.631 → ≈0.74;
  tabloda 0.726). My-FuzzyCDF öznel %50: 0.622 ve 0.442 → ≈0.52; tabloda 0.587. Makro/mikro ortalama belirtilmemiştir.
- FuzzyCDF ve My-FuzzyCDF nesnel F1 değerleri %60'tan %70'e düşer (0.557→0.537; 0.573→0.561): makalenin kendi
  sonuçları da eğitim oranıyla monoton değildir.
- Makalenin DINA/FuzzyCDF mutlak sayıları (PRE 0.60-0.74) bizim FrcSub sonuçlarımızdan (PRE ≈ 0.82-0.85) çok farklıdır;
  veri ve protokol farkı nedeniyle mutlak sayılar karşılaştırılmaz, yalnızca eğilimler.
