# Veri

Ham veri lisans belirsizliği nedeniyle repoya konmaz (`data/raw/` git tarafından yok sayılır).

## FrcSub (gerçek veri)
Kesir çıkarma testi: 536 öğrenci x 20 soru, 8 beceri, ikili cevaplar. Yalnızca objektif sorular
içerir; zaman ve süre bilgisi **yoktur**.

```bash
python3 -m src.data_frcsub   # indirir, SHA-256 doğrular, FrcSub'ı açar, boyutları kontrol eder
```
Betik yalnızca `FrcSub` klasörünü açar. Paketteki Math1/Math2 verileri kullanılmaz
(kullanım koşulları ayrıdır). RAR açmak için macOS'te hazır gelen `bsdtar` kullanılır.

**Kaynak:** R. Wu, Q. Liu, Y. Liu, E. Chen, Y. Su, Z. Chen, G. Hu, "Cognitive Modelling for
Predicting Examinee Performance," IJCAI 2015 (veri paketinin `TermsOfUse.txt` dosyasında bu
şekilde verilmiştir; kesin künye rapor aşamasında doğrulanacaktır).
Adres: http://staff.ustc.edu.cn/~qiliuql/data/math2015.rar

## Sentetik veri
Makalenin yapısını taklit eder (1000 öğrenci, 50 soru, 25 beceri, kayıt zamanı, cevap süresi).
**Açıkça sentetiktir**; gerçek veri gibi sunulmaz.

```bash
python3 -m src.data_synthetic   # özet istatistikleri yazdırır
```
```python
from src.data_synthetic import generate
d = generate(seed=0)   # aynı seed -> aynı veri
```
Üretim hikâyesi ve döngüsellik uyarısı [src/data_synthetic.py](../src/data_synthetic.py)
dosyasının başındaki açıklamada yer alır.
