# PROJECT BRIEF — Bulanık Mantık Dönem Projesi

> Bu dosya, Claude (sohbet) ile yapılan planlamanın Claude Code'a devridir.
> Repo kökünde `docs/PROJECT_BRIEF.md` olarak tutulacak. Her oturumda ilk okunacak dosya budur.

## 0. Künye
- Öğrenci: Musa Semih BÜYÜKÖZKAN, no: 1306230099 (makale no: 38)
- Ders: Bulanık Mantık, dönem projesi. **Teslim: 15.11.2026** (uzatıldı). Öğrenci işi 3–4 günde bitirmek istiyor.
- Referans makale: Y. Zhang & Y. Wang, "A personalized recommendation algorithm for English exercises incorporating fuzzy cognitive models and multiple attention mechanisms", Scientific Reports 15:11531 (2025), DOI 10.1038/s41598-025-96489-3. Lisans: CC BY-NC-ND (makale PDF'ini ve şekillerini repoya KOYMA, şemaları kendimiz çizeriz).
- Yönerge puanları: Sunum 20 / Yeniden gerçekleştirme 30 / Özgün katkı 30 / Rapor+GitHub 20.

## 1. Claude Code için çalışma kuralları (ÖNEMLİ)
1. Öğrenci bulanık mantığı henüz bilmiyor, lisans seviyesinde bir proje hedefliyoruz. Kod **sade, küçük fonksiyonlu, bol yorumlu** olsun. Gereksiz soyutlama, ağır framework yok.
2. Her modül için: (a) ne işe yarar, (b) hangi formüle karşılık gelir (makale Eş. no), (c) küçük bir test. Her modülden sonra öğrenciye 3–5 cümlelik "bu kod ne yapıyor, neden" açıklaması ver.
3. **Hiçbir sayıyı uydurma.** Rapora ve sunuma giden her tablo/grafik, repodaki betiğin çıktısından gelmeli. Başarısız ya da beklenmedik sonucu olduğu gibi raporla (yönerge bunu açıkça kabul ediyor).
4. Harici koddan (EduCDM vb.) bir şey alınırsa kaynağı ve lisansı `references/README.md`'de ve kodda belirt; kendi kodumuzdan ayır.
5. Gizli bilgi (API anahtarı, token) repoya girmesin. `.gitignore` kullan.
6. Küçük, anlamlı commit'ler at (yönerge commit geçmişini değerlendiriyor). Tek seferde yükleme yapma.
7. Yapay zekâ kullanımını `docs/ai_usage_log.md` dosyasına işle (kod geliştirme, hata ayıklama, kavram açıklama vb.). Raporda "Generative AI Usage Disclosure" bölümü bundan beslenecek.
8. Kaynakça: yalnızca doğrulanmış kaynaklar. Makalenin kendi kaynakçası **kopyalanmayacak** (bkz. §6).
9. Python 3, macOS, VS Code. Bağımlılıklar `requirements.txt`'te sürümleriyle.

## 2. Makale özeti (MFCD-MAKT)
Amaç: öğrencinin henüz çözmediği alıştırmadaki başarısını tahmin edip alıştırma önermek.
Üç parça:
1. **MFCD** (geliştirilmiş fuzzy cognitive diagnosis, FuzzyCDF tabanlı)
   - Eş. 1: bilgi noktası k için yeterlilik = üyelik derecesi
     `a_ik = μ_k(i) = 1 / (1 + exp(-1.7 · α_k · (θ_i − β_k)))`
   - Objektif soru: bağlayıcı (fuzzy AND, ∩) → Eş. 3/5; subjektif soru: telafi edici (fuzzy OR, ∪) → Eş. 6.
   - Üç yeni faktör: (i) zaman ağırlığı `ω = (T_m − Δt)/T_m` (Eş. 2, 4); (ii) cevap süresi ağırlığı `ωC = (maxC_n − C_mn) · R_mn` (Eş. 7); (iii) bilgi açığı `l_ijk = μ_jk (1 − a_ik)`, `η_ij = 1 − max_k l_ijk` (Eş. 8–9).
   - Parametreler (θ, Ψ, Ω; Ω = slip s, guess g, önem μ) MCMC / Metropolis-Hastings ile kestirilir (Eş. 11–16).
2. **MAKT**: 3 başlı dikkat ile alıştırma benzerliği (Eş. 17–19), BiGRU ile bilgi durumu (Eş. 20–21), tahmin ağı + BCE + Adam (Eş. 22–25).
3. Öneri: tahmin edilen puan ve zorluğa göre.

Makale deneyi: 1000 öğrenci × 50 soru (38 okuma/objektif, 6 yazma, 6 çeviri), 25 bilişsel nitelik (A1–A25). Eğitim oranı %40–90; metrikler PRE, Recall, F1 (doğru cevap = pozitif), MAE, RMSE. Karşılaştırılanlar: DINA, FuzzyCDF, My-FuzzyCDF, DKT, C&RM-MAKT. **Veri kapalı** (yazardan istek üzerine).

## 3. Kapsam kararları

### 3.1 MRR (yönergedeki minimum yeniden gerçekleştirme)
Bulanık çekirdek **MFCD-core** gerçekleştirilecek. MAKT (BiGRU + dikkat) kapsam DIŞI; raporda "bilinçli sınırlama" olarak açıklanacak (yönerge §5.1 kapsamlı makalelerde ana bulanık bileşene izin veriyor). Zaman kalırsa isteğe bağlı küçük bir PyTorch dikkat modülü eklenebilir; ZORUNLU DEĞİL.

MFCD-core içeriği:
- Üyelik fonksiyonu (Eş. 1, lojistik/sigmoid tipi).
- Fuzzy operatörler: objektif için t-norm (AND), subjektif için t-conorm (OR). Operatör **parametre** olacak (bkz. katkı).
- Bilgi açığı (Eş. 8–9).
- Slip/guess ile cevap olasılığı: `P(r_ij=1) = (1 − s_j)·η_ij + g_j·(1 − η_ij)`.
- Zaman/süre ağırlıkları (Eş. 2, 7) — yalnızca zaman/süre verisi olan veride (sentetik) etkin.
- Parametre kestirimi: numpy ile basit random-walk Metropolis-Hastings (Eş. 14–16 mantığı). Yakınsama sorunu olursa gradyan tabanlı MAP'e geç ve bunu raporda belirt.

### 3.2 Veri kararı (araştırma sonucu)
Hem Q-matrisi hem zaman damgası hem cevap süresi hem de İngilizce içerik taşıyan açık bir CDM seti bulamadık. Karar:
- **Gerçek veri: FrcSub** (kesir çıkarma, 536 öğrenci × 20 soru, 8 beceri, ikili 0/1, yalnızca objektif, zaman bilgisi yok). Kaynak olarak bildirilen adres: `http://staff.ustc.edu.cn/~qiliuql/data/math2015.rar` (EduData kütüphanesi de erişim sağlıyor). **İndirdikten sonra boyutları doğrula** (536×20, K=8); farklı çıkarsa dur ve öğrenciye bildir. Ham veri lisansı belirsizse repoya koyma, indirme betiği + README talimatı yaz.
- **Sentetik veri**: makalenin yapısını taklit eder (1000 öğrenci, 50 soru: 38 objektif + 6 + 6 subjektif, K=25, kayıt/cevap zamanları, cevap süreleri). Amaç, gerçek veride test edilemeyen zaman/süre/açık faktörlerini ve parametre geri kazanımını (bilinen gerçek θ ile) sınamak. Raporda **açıkça sentetik** denecek.
- **İsteğe bağlı: Math1/Math2** (objektif+subjektif, [0,1] skorlar). Aynı paketin sayfası bunları "private" olarak tanımlıyor ve Terms of Use dosyası var → kullanmadan önce `TermsOfUse.txt` okunacak ve öğrenciye sorulacak. İzin belirsizse KULLANMA.

### 3.3 Baseline'lar
- DINA (klasik, ayrık), IRT-2PL (referans), **FuzzyCDF-orijinal** (üç faktör kapalı, min/max). İsteğe bağlı: EduCDM'deki hazır uygulamalarla çapraz kontrol (kaynak göstererek). DKT/C&RM-MAKT yeniden gerçekleştirilmeyecek; makaledeki sayılarla yalnızca eğilim karşılaştırması yapılacak.

### 3.4 Özgün katkı
**K1 — Fuzzy operatör karşılaştırması:** t-norm/t-conorm aileleri: Gödel (min/max), çarpım (a·b / a+b−ab), Łukasiewicz (max(0,a+b−1) / min(1,a+b)). Hem objektif hem subjektif katmanda.
**K2 — Faktör ablation'ı:** zaman, süre, bilgi açığı faktörlerini tek tek/ikili aç-kapa (makale hepsini birlikte sunuyor, ayrı ayrı etkisini göstermiyor).
**K3 (zaman kalırsa) — Gürültü dayanıklılığı:** etiketlerin %5/10/20'si çevrilerek.
Başarı ölçütü: baseline'a karşı PRE/Recall/F1/MAE/RMSE ve çalışma süresi (yönerge §6.2 tablosu). İyileşme zorunlu değil; yorum zorunlu.

**Önemli matematiksel not (rapora ve savunmaya girecek):** μ_jk ikili (=q_jk) iken `η_ij = 1 − max_k q_jk(1 − a_ik) = min_{k: q=1} a_ik`; yani bilgi açığı formülü Gödel t-norm'una (min) indirgenir. FrcSub'da (ikili Q) açık faktörü min operatöründen ayırt edilemez. Açığın anlamlı olması için μ'nün [0,1] aralığında derecelenmesi gerekir (Ω içinde öğrenilen μ, q ile maskelenmiş). Sentetik veride μ dereceli üretilecek. Bu bulgu K2'nin bir parçası olarak raporlanacak.

**Sentetik veri uyarısı:** Sentetik veriyi zaman/süre etkisiyle üretirsek bu faktörleri içeren model doğal olarak kazanır (döngüsel). Raporda bunun "faktörlerin geri kazanılabilirliğini" gösterdiğini, gerçek dünya faydasını kanıtlamadığını açıkça yaz.

## 4. Deney tasarımı
- Bölme: her öğrencinin kayıtlarının eğitim oranı p ∈ {40,50,60,70,80,90}% rastgele seçilir, kalanı test (makale Şekil 3 ile uyumlu). Tablo benzeri raporlama için 50/60/70.
- Tekrar: ≥10 farklı seed; ortalama ± standart sapma. Seed'ler sabit ve kayıtlı.
- Metrikler: PRE, Recall, F1 (doğru=pozitif, eşik 0.5), MAE, RMSE, eğitim süresi.
- Makale ile karşılaştırma: sayılar birebir eşleşmeyecek (veri kapalı). Karşılaştırılacak **eğilimler**: (i) model sıralaması DINA < FuzzyCDF < geliştirilmiş MFCD mi, (ii) eğitim oranı arttıkça iyileşme, (iii) subjektif < objektif başarı (yalnızca subjektif içeren veride).
- E1: FrcSub üzerinde baseline'lar + MFCD-core. E2: sentetik veride aynısı + ablation (K2). E3: operatör karşılaştırması (K1) iki veride. E4: süre ölçümü.

## 5. Klasör yapısı (yönergeye uygun)
```
fuzzy-logic-project/
├── README.md  LICENSE  requirements.txt  .gitignore
├── docs/PROJECT_BRIEF.md  docs/ai_usage_log.md  docs/assumptions.md
├── src/            # fuzzy_ops.py, memberships.py, mfcd.py, mcmc.py, baselines.py, data_*.py, metrics.py
├── data/           # indirme betiği + README (ham veri lisansa göre)
├── experiments/baseline/   experiments/proposed/
├── results/figures/  results/tables/
├── presentation/presentation.pdf   report/report.pdf
└── references/README.md
```
README başlıkları (yönerge §10): Project Title, Student Information, Original Paper, Problem Description, Method, Fuzzy System Architecture, Dataset, Installation, Requirements, How to Run, Reproduction Experiments, Proposed Extension, Experimental Results, Limitations, References.

## 6. Makale hakkında bilinen sorunlar (varsayım günlüğü: `docs/assumptions.md` buraya taşınacak)
- Eş. 2 ile Eş. 4 aynı; Eş. 3 ile Eş. 5 aynı. Zaman ağırlığının η'ya **nasıl** uygulandığı belirsiz. **Varsayım:** zaman ağırlığı gözlemin olabilirlik (likelihood) ağırlığıdır ("eski veri modele daha az katkı yapar" ifadesine uygun); alternatif: η ile çarpım. Varsayılan = ağırlıklı likelihood. Bunu rapora yaz.
- Eş. 7 normalize değil. **Varsayım (bizim işletmemiz):** `ωC` / maxC_n ile [0,1]'e ölçekle; gözlem ağırlığı `w = ω_time · (1 + λ·ωC)`, λ=1 varsayılan hiperparametre. Yanlış cevapta ωC=0 (makale ile uyumlu). Bunun makaleden değil bizim yorumumuzdan geldiğini rapora yaz.
- Eş. 1'de `a` hem ayırt edicilik hem yeterlilik olarak kullanılmış; biz α_k (ayırt edicilik), β_k (zorluk), a_ik (yeterlilik) kullanıyoruz.
- Eş. 12–13'te işaret/gösterim hataları ("−P"); standart Bernoulli olabilirliği kullanılacak. Eş. 29–30'da P ve r tanımları ters.
- Veri tasarımında skor "ikili" deniyor ama Eş. 7 kesirli puan oranı kullanıyor.
- Önerilen model için hiperparametreler verilmemiş (yalnızca baseline'lar: epoch 100, batch 16, lr 0.002).
- Tablo 4: metin "her tipte ve oranda en iyi" diyor, fakat subjektif %50 Recall'da C&RM-MAKT (0.532) MFCD-MAKT'tan (0.525) yüksek; ayrıca PRE ve Recall'dan çıkan F1 tablodaki F1 ile birebir tutmuyor (ör. 0.894 ve 0.631 → ≈0.74, tabloda 0.726), makro/mikro ortalama belirtilmemiş.
- Öneri kalitesi (ör. nDCG) ölçülmemiş; yalnızca performans tahmini.
- Kaynakça güvenilmez: 25 numaralı kaynak 17'nin kopyası; DKT için gösterilen 26–27 IRT makaleleri; 14 numara dalga ışını izleme üzerine; 1–4 numaralar müzik/turizm/haber. **Makalenin kaynakçası kopyalanmayacak**; her kaynak (DOI/yayıncı sayfası) doğrulanarak eklenecek. Aday: Wu vd., FuzzyCDF (ACM TIST); Piech vd., DKT (NeurIPS 2015); DINA için de Junker & Sijtsma vb. — hepsi DOĞRULANMADAN rapora girmez.

## 7. Aşamalar (öğrenci sınav dönemi, esnek)
- **A1** Repo iskeleti, requirements, README taslağı, `docs/` dosyaları. (commit)
- **A2** Veri: FrcSub indirme+doğrulama betiği, sentetik veri üreteci + testler.
- **A3** `fuzzy_ops`, `memberships`, `mfcd` çekirdeği + birim testler (küçük elle hesaplanmış örnekler).
- **A4** Metropolis-Hastings kestirimi; sentetikte parametre geri kazanımı kontrolü (θ korelasyonu).
- **A5** Baseline'lar + E1.
- **A6** K1 ve K2 deneyleri, tablolar, grafikler (kendi çizimlerimiz).
- **A7** Rapor iskeleti (İngilizce başlıklar, IEEE kaynakça, 8–12 sayfa) + AI Usage Disclosure.
- **A8** README cilası, temiz kurulumla yeniden çalıştırma testi (clone → install → run).
Sunum metni ve Canva taslağı sohbet tarafında (Claude ile) yazılacak; deney sonuçları hazır olunca öğrenci buraya döner.

## 8. Savunma için kod açıklama kontrol listesi
Öğrenci şunları kendi sözleriyle açıklayabilmeli (her biri için kodda işaretli bir yer olsun):
1. Fuzzy küme ve üyelik fonksiyonu nedir? Eş. 1 neden "üyelik derecesi" sayılır? (sigmoid, θ−β)
2. t-norm (AND) ve t-conorm (OR): min/max, çarpım, Łukasiewicz farkı; neden objektif=AND, subjektif=OR?
3. Bilgi açığı formülü ve min'e indirgenme gözlemi.
4. Slip ve guess nedir, P(r=1) formülü.
5. Zaman ve süre ağırlıklarının mantığı; bizim varsayımlarımız.
6. Metropolis-Hastings: öneri dağılımı, kabul olasılığı (Eş. 14), neden çalışır.
7. Eğitim/test bölmesi, PRE/Recall/F1 farkı, eşik.
8. Neden MAKT kapsam dışı? Neden FrcSub + sentetik? Sentetik verinin döngüsellik sınırı.
9. Katkının gerekçesi ve sonuçların yorumu (iyileşme olmasa da).
10. Sınırlılıklar: kapalı veri, ikili Q, sentetik veri, tek veri seti, MCMC yakınsaması.
