# Rapor için Çalışma Notları

Sonda rapora ve sunuma taşınacak kararlar, gözlemler ve ayrıntılar. Sayılar buraya
yalnızca betik çıktısından yazılır.

## Tasarım kararları
- **Birleşik formülasyon (bizim):** `ã = 1 − μ(1 − a)` (VE katmanı) ve `ã = μ·a` (VEYA katmanı),
  ardından seçilen t-norm / t-conorm ile birleştirilir. Gödel + VE durumunda bu, makaledeki
  bilgi açığı formülü (Eş. 8-9) ile birebir aynıdır. Böylece operatör karşılaştırması (K1) ve
  açık faktörü (K2) tek bir çerçevede ele alınır.
- **Açık faktörü kapatmak** = μ'yü ikili Q'ya eşitlemek. İkili Q'da bilgi açığı min'e iner;
  FrcSub'da (ikili Q) açık faktörü ayırt edilemez (matematiksel not, assumptions.md).
- Zaman/süre ağırlıkları olabilirlik ağırlığıdır (makale belirsiz; varsayım).
- Eş. 7 normalizasyonu bize aittir: `ωC = (maxC_j − C_ij)/maxC_j · R_ij`, `w = ω_time(1 + λωC)`, λ = 1.
- Zaman ağırlığı Δt = T_m olan kayıtta tam 0 olur (formül gereği); alt sınır konmadı.

## Savunma için hatırlatmalar
- Üyelik fonksiyonu: `src/memberships.py`; operatörler: `src/fuzzy_ops.py`; birleşim: `src/mfcd.py`.
- Elle hesaplı testler: `tests/test_fuzzy_ops.py`, `tests/test_mfcd.py`.

## Teknik notlar
- Python 3.10.11 (sistem), paketler `requirements.txt` ile; venv kullanılmadı.
- Bir yan not: `.venv` Python 3.13 ile açılınca pinli numpy/scipy derlenemedi (3.13 için uygun
  paket yok). Rapora girmeyecek, ama temiz kurulum testinde (A8) Python 3.10 belirtilmeli.

## A4: Metropolis-Hastings (MCMC)
- Uygulama: `src/mcmc.py`. Random-walk MH; θ ve (s, g, μ) vektörel, α/β beceri beceri güncellenir.
- Önseller bizim seçimimiz: θ~N(0,1), β~N(0,1), log α~N(0,0.5²), s,g~U(0,0.4), μ~U(0,1].
- Zincir: 1500 adım, ilk 700 burn-in, her 10. örnek (tahmin = posterior ortalama P).
- Adım boyları kabul oranı ~0.25-0.55 olacak şekilde elle ayarlandı (θ 0.9, α/β 0.25, μ 0.15).
- **Yakınsama sorunu çıkmadı**, gradyan tabanlı MAP'e geçmek gerekmedi.
- Pilot sonuçlar (sentetik, seed 0, %60 eğitim; kaynak: `experiments/proposed/a4_mh_recovery.py`,
  `results/tables/a4_mh_recovery.csv`):
  θ korelasyonu 0.863 (toplam skor 0.837); test doğruluğu 0.664 (gerçek parametreler 0.674,
  çoğunluk sınıfı 0.532); log-olabilirlik 300-500 ve son 800 adımda aynı düzeyde (-17947 / -17949).
- Yorum: Test doğruluğu gerçek parametreli "oracle" tahmine yakın; kalan fark ikili cevaplardaki
  slip/guess gürültüsünden gelir. Tek seed; makalede istenen ≥10 seed deneyi A5-A6'da yapılacak.
- Süre: ~65 sn / zincir (1500 adım). Deney ızgarası bunu hesaba katarak kurulacak.
- Not: üretecin zaman kayması (drift) modele verilmediği için ağırlıksız model bunu görmez;
  zaman ağırlığının etkisi K2 ablation'ında ölçülecek.

## A5 / E1: FrcSub üzerinde karşılaştırma (GÜNCEL, 3 başlangıçlı MH)
Kaynak: `experiments/baseline/e1_frcsub.py` -> `results/tables/e1_frcsub_{raw,summary}.csv`, `e1_frcsub_table.md`;
grafik: `results/figures/e1_frcsub_f1_rmse.png`. 4 model x 6 eğitim oranı x 10 seed (0-9). MH 1500 adım (700 burn-in).
**FrcSub'da iki MH modeli de 3 farklı başlangıçla çalışır; EĞİTİM olabilirliği en yüksek zincir seçilir** (test'e
bakılmaz). Süreler 3 başlangıcın toplamıdır. Gerekçe aşağıdaki yakınsama bölümünde.

**Geçmiş (dürüstlük notu):** İlk E1 koşusu tek zincirle yapılmıştı; orada MFCD-core, FuzzyCDF-orig'in F1'inin ~0.008
altındaydı. Yakınsama incelemesi MFCD-core'un FrcSub'da çok modlu olduğunu gösterdi; tek zincir bazen kötü çukura
düşüyor. 3 başlangıçla fark ~0.003'e indi. Tek zincirli sonuçlar nihai sonuç değildir.

**Bulgular (%60 eğitim, 10 seed ortalaması):**
- F1: FuzzyCDF-orig 0.849 > MFCD-core 0.845 > IRT-2PL 0.834 > DINA 0.813.
- RMSE: FuzzyCDF-orig 0.352 = MFCD-core 0.352 < IRT-2PL 0.363 < DINA 0.372. MAE'de IRT-2PL en iyi (0.241).
- PRE'de DINA en yüksek (0.850); Recall'da en düşük (0.780). DINA "doğru" demekte temkinli.
- Süre: IRT 0.04 sn, DINA 1.7 sn, FuzzyCDF-orig 27.9 sn, MFCD-core 31.5 sn (3 başlangıç dahil; tek zincir ~9-10 sn).
- Eşleştirilmiş fark (MFCD-core − FuzzyCDF-orig, aynı seed ve bölme), F1: -0.003 ... -0.005 (her oranda 10 seed'in
  0-2'sinde MFCD önde); RMSE: ≈ 0 (±0.001; 10 seed'in 2-6'sında MFCD önde). Yani F1'de küçük ama tutarlı bir
  dezavantaj, RMSE'de fark yok.

**Makaleyle karşılaştırma (yalnızca eğilimler; Tablo 4 objektif sorular):**
- Eğitim oranı arttıkça iyileşme: **uyuşuyor** (bizim tüm modellerimizde F1 artıyor, RMSE azalıyor; fark küçük
  noktalarda seed gürültüsü içinde).
- DINA < FuzzyCDF (F1): **uyuşuyor** (0.813 < 0.849 @%60; makalede 0.461 < 0.557).
- DINA < FuzzyCDF (PRE): **uyuşmuyor.** Makalede DINA'nın PRE'si en düşük; bizde DINA'nın PRE'si en yüksek
  (0.850 vs 0.823). Bizim DINA'mız Recall'ı feda ederek kesinliği artırıyor.
- FuzzyCDF < geliştirilmiş model (MFCD): **uyuşmuyor.** MFCD-core, FuzzyCDF-orig'i geçmiyor. Nedenler:
  (i) FrcSub'da zaman/süre bilgisi yok, makalenin üç faktöründen ikisi devre dışı; (ii) Q ikili, bilgi açığı
  formülü zaten min'e iner, öğrenilen μ ek serbestlik ama ek bilgi değil; (iii) makalenin büyük sıçraması MAKT'tan
  geliyor (kapsam dışı); (iv) μ öğrenmek çok modlu bir posterior yaratıyor, 3 başlangıç yetmeyebilir.
- Mutlak sayılar karşılaştırılamaz (veri kapalı; makalede DINA PRE 0.60-0.71, bizde ≈0.85).
- Subjektif < objektif başarı: FrcSub'da subjektif soru yok, **test edilemedi** (sentetikte test edilebilir ama
  üreteç parametrelerinin sonucu olur, bulgu sayılmaz).

**Ek notlar:**
- DINA kendi gerçekleştirmemiz (Gibbs); makaledeki "higher-order DINA" ile birebir aynı olmayabilir.
- IRT-2PL: MAP (L-BFGS); gradyan sayısal olarak doğrulandı.
- DINA zincirleri tutarlı (4 seed, F1 aralığı ≈0.002; `results/tables/dina_chain_check.csv`).

## MCMC yakınsama kanıtı
Kaynak: `experiments/proposed/mcmc_convergence.py` -> `results/tables/mcmc_convergence.csv`,
`mcmc_chain_metrics.csv`, grafik `results/figures/mcmc_trace.png`. Kurulum: 4 zincir x {FrcSub, sentetik} x
{FuzzyCDF-orig, MFCD-core}, %60 eğitim, 3000 adım (1500 burn-in), iki başlangıç türü ("dağınık" ve "üretim").

**Sentetik veri: yakınsıyor (tahmin açısından).** Log-olabilirlik R-hat 1.000-1.010; θ için R-hat en fazla 1.027;
dört zincirin test F1 değerleri arasında fark ≤ 0.002. Ama α, β, slip, guess ve özellikle μ için R-hat > 1.1 olan
parametreler var (MFCD-core, dağınık: α 7/25, β 9/25, slip 12/50, guess 9/50, μ 60/129). Bu parametreler zayıf
belirlenmiş (farklı değerler aynı tahmini veriyor), yani **parametre yorumu güvenilmez, tahmin güvenilir**.
Rapora "bireysel μ, α, β değerlerini yorumlamıyoruz" cümlesi girmeli.

**FrcSub: yakınsama sorunu var.** Dağınık başlangıçta log-olabilirlik R-hat 2.04 (FuzzyCDF-orig) ve 1.68 (MFCD-core);
bir zincir (zincir 2) uzun süre daha kötü bir platoda kalıyor (log-olabilirlik ≈ -2450 vs ≈ -2300). Üretim
başlangıcında FuzzyCDF-orig iyi (R-hat 1.05, F1 aralığı 0.847-0.848), MFCD-core kötü (R-hat 2.16; 4 zincirden
biri kötü çukurda, F1 0.830 vs 0.844-0.847). Bu yüzden FrcSub'da 3 başlangıç + en iyi eğitim olabilirliği kullanılır.

**Çok modluluğun boyutu (E1, 120 MH koşusu):** 3 başlangıcın eğitim log-olabilirlik ortalamaları arasındaki fark > 30
olan koşu sayısı: MFCD-core 39/60, FuzzyCDF-orig 5/60. Yani "yakınsamayan" (başlangıca duyarlı) koşu oranı MFCD-core
için yüksek. 3 başlangıç bunu azaltır ama garanti etmez.

**Sürüklenme ölçüsü (`ll_surukleme`) kısıtı:** Burn-in sonrası izin iki yarısının ortalama farkı / std. Seçilen zincirde
120 koşunun 46'sında > 0.5. Ancak çukurda *yatay* giden zinciri yakalamaz (üretim başlangıcında çukurdaki zincirin
ölçüsü 0.41 idi). Bu yüzden asıl kanıt R-hat ve zincirler arası düzey farkıdır; sürüklenme yardımcı bir göstergedir.

## K1 (ANA KATKI): bulanık operatör karşılaştırması
Kaynak: `experiments/proposed/k1_operators.py` -> `results/tables/k1_{raw,summary,paired}.csv`; grafik:
`results/figures/k1_operators.png` (`plot_k1.py`). Model: μ = Q (açık faktörü kapalı), zaman/süre kapalı; yalnızca operatör
değişir. Her ayar sıfırdan eğitilir; aynı seed, aynı eğitim/test bölmesi, aynı hiperparametre. FrcSub: yalnızca VE
(3 aile, oranlar 0.5/0.6/0.7, FrcSub'da 3 başlangıç); sentetik (zaman/süre etkili): VE ve VEYA, oran 0.6, 5 ayar.
Karşılaştırma **eşleştirilmiş fark** ile (aynı seed'deki Gödel koşusuyla fark; seed'ler arası veri değişkenliği çıkar).

**FrcSub (gerçek veri):** operatörler pratikte aynı. %60'ta F1: Gödel 0.849, çarpım 0.848, Łukasiewicz 0.847;
RMSE 0.352 / 0.352 / 0.353. Gödel'e göre F1 farkları -0.0007 ... -0.0019 (std 0.0014-0.0032; hata çubukları çoğunlukla
sıfırı kesiyor; diğer operatörün Gödel'i geçtiği seed sayısı 1-5/10). Yani fark seed gürültüsü düzeyinde.
**Sentetik (oran 0.6):** F1: Gödel/Gödel 0.618; VEYA=çarpım 0.616 (-0.0019); VEYA=Łuk. 0.614 (-0.0042);
VE=çarpım 0.613 (-0.0050); VE=Łuk. 0.610 (-0.0076). RMSE farkları +0.0005 ... +0.0015 (kötüleşme). Gödel dışındaki ayarlar
10 seed'in 10'unda (VEYA=çarpım için 9'unda) Gödel'den kötü. Sıralama her iki katmanda **Gödel > çarpım > Łukasiewicz**;
VE katmanı VEYA katmanından daha duyarlı.

**Yorum ve sınırlar (rapora girecek):**
- **Döngüsellik:** Sentetik veriyi Gödel (min / ağırlıklı max) ile ürettiğimiz için Gödel'in kazanması beklenen bir
  sonuçtur; operatörlerin gerçek dünyadaki üstünlüğünü göstermez. Gerçek veri (FrcSub) operatörler arasında anlamlı
  fark bulamadı.
- Mutlak farklar küçük (F1 ≤ 0.008, RMSE ≤ 0.0015): pratik etki sınırlı.
- Sezgisel açıklama (test edilmemiş hipotez): Łukasiewicz VE'si, birkaç beceri gerektiren sorularda toplamdan (n-1)
  çıkardığı için η'yı sık sık tam 0'a indirir; bu bölgede olasılık sabit (≈ guess) kalır ve öğrenciler ayırt edilemez.
  Gödel'de ise yalnızca en zayıf beceri belirleyicidir.
- FrcSub'da yalnızca VE test edilebildi (subjektif soru yok). VEYA yalnızca sentetikte.
- **Süre sütunları karşılaştırılamaz:** 10 süreç aynı anda koştuğundan koşular birbirini farklı oranda yavaşlattı (ör. aynı
  iş 105 sn ile 302 sn arasında ölçüldü). Süre karşılaştırması (E4) ayrı, tek süreçli ve ayrı bir koşuyla yapılacak.
- Log-olabilirlik 0'a gitmedi: Łukasiewicz'de η = 0 olsa bile P ≥ guess > 0 (testle doğrulandı, `tests/test_a6.py`).

## K2: faktör ablation'ı (yalnızca sentetik) ve NEGATİF KONTROL
Kaynak: `experiments/proposed/k2_ablation.py` -> `results/tables/k2_{raw,summary,paired}.csv`; grafik
`results/figures/k2_ablation.png` (`plot_k2.py`). 7 ayar + IRT-2PL + DINA, iki veri varyantı ("synthetic": zaman/süre
etkili; "synthetic_null": negatif kontrol, zaman ve süre cevaptan bağımsız), oran 0.6, 10 seed, tek zincirli MH
(sentetikte zincirler tahmin açısından tutarlı; bkz. yakınsama bölümü). Karşılaştırma "hiçbiri"ye göre eşleştirilmiş fark.
Yeniden koşu birebir aynı sonucu verdi (F1/RMSE farkı 0.0, 180 satır). AUC ve kalibrasyon sütunları ikinci koşuda eklendi.

**Ana bulgu (olumsuz ve rapora olduğu gibi girecek): hiçbir faktör AUC'yi (eşikten bağımsız sıralama başarısı) artırmadı.**
"hiçbiri"ye göre AUC farkı (etkili sentetik): +bilgi açığı -0.0005; +zaman (olab.) -0.0072; +zaman (Eş.3) -0.0319;
+bilgi açığı +zaman -0.0085; +süre -0.0019; hepsi -0.0066. Hiçbirinde 10 seed'in çoğunda iyileşme yok.

**F1'deki "kazançlar" kalibrasyon kaymasıdır:**
- "+süre (bizim yorum)" F1'i +0.053 artırıyor (etkili) ve +0.031 artırıyor (NEGATİF KONTROL, süre bilgisizken bile).
  Aynı anda Recall 0.59 -> 0.79, PRE 0.65 -> 0.58, RMSE kötüleşiyor (+0.011) ve ortalama tahmin 0.58'e çıkıyor (gerçek
  doğru oranı 0.48). Açıklama: ωC yalnızca DOĞRU cevaplara bonus veriyor (yanlışta ωC = 0), olabilirlik doğru cevaplara
  daha fazla ağırlık verince model "doğru" demeye kayıyor; doğru sınıf pozitif olduğu için F1 yükseliyor. Bu, süre bilgisinden
  gelen bir fayda DEĞİL, bir eşik/yanlılık yan etkisi. Negatif kontrol bunu yakaladı.
- "+zaman (olabilirlik)": F1 +0.0086 (etkili, 10/10 seed) ama kontrol veride -0.0022; AUC her iki veride ≈ -0.007/-0.008 ve
  RMSE kötüleşiyor (+0.003). F1 artışı küçük ve ortalama tahmin 0.481 -> 0.494'e kayıyor; yani sıralama iyileşmedi.
- "+zaman (Eş.3 çarpımı)" (makalenin yazımı, bizim uygulamamızla): F1 -0.104 (etkili) ve -0.128 (kontrol); AUC -0.032 / -0.058;
  RMSE +0.011 / +0.019. Yeterliliği (T-Δt)/T ile çarpmak tahminleri aşağı çekiyor (ort. P 0.454). Makaledeki yazımın bizim
  uygulamamız bu şekilde işe yaramadı; makalenin gerçekte ne yaptığını bilmiyoruz (belirsizlik notu: assumptions.md).
- "+bilgi açığı" (dereceli μ): F1 +0.0033 (etkili, 10/10 seed) ve +0.0012 (kontrol, 7/10); AUC ve RMSE ≈ değişmez. Etki
  çok küçük. Veride μ dereceli üretildiği halde MH ile μ'yü öğrenmek zor (çok parametre, zayıf belirlenmişlik).
- Baseline'lar: IRT-2PL, MFCD varyantlarıyla aynı düzeyde ya da daha iyi MAE/RMSE'ye sahip (MAE 0.425 vs 0.435);
  DINA açık farkla en kötü (F1 -0.138, AUC -0.061), makaledeki DINA < FuzzyCDF eğilimiyle uyumlu.

**Yorum (sade):** Veriyi zaman ve süre etkisiyle ürettik, ama bu bilgileri "gözlem ağırlığı" olarak kullanmak (veya
Eş. 3 gibi çarpmak) modelin sıralama başarısını artırmadı. Olası nedenler (test edilmemiş): üretecimizin zaman/süre
etkileri zayıf olabilir; ağırlıklandırma bu bilgiyi kullanmanın zayıf bir yolu olabilir (süreyi cevabın *olasılığına*
bağlayan bir model daha uygun olurdu); tek zincirli MH gürültüsü. Bu yüzden sonuç "faktörler geri kazanılabiliyor" diye
okunmamalı; **bizim uygulamamızda geri kazanılamadılar**. F1 kazançları AUC ve kalibrasyonla çelişiyor.
**Rapor dili:** F1/PRE/Recall tek başına eşik ve yanlılık yüzünden yanıltabilir; bu yüzden AUC ve RMSE birlikte raporlanır.
**Sentetik sonuçlar** "faktörler geri kazanılabiliyor mu" sorusunu cevaplar, "gerçek hayatta işe yarar mı" sorusunu cevaplamaz.

**Yakınsama:** K2'nin 140 MH koşusunda sürüklenme ölçüsü ortalaması 0.15-0.78 (en yüksek: Eş.3 çarpımı, 0.50/0.78).
Eş.3 ayarı en kötü yakınsayan ayardı.
**Eksikler:** K2 yalnızca oran 0.6'da; "+süre" ayrı bir normalizasyon (λ=1) ile denendi, farklı λ denenmedi.
