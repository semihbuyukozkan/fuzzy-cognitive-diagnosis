# Savunma hazırlığı: olası sorular ve kısa cevaplar

Yönerge §14: canlı demoda sistemin ürettiği sonucu teknik olarak açıklamanız ve şu başlıklarda soru alacaksınız: fuzzy kuralların işlevi,
üyelik fonksiyonu seçimi, parametrelerin etkisi, algoritmalar, değişikliklerin gerekçesi, sonuçların nedenleri, kodun bölümleri, sınırlılıklar.
Cevapları kendi cümlelerinizle söyleyin; aşağıdakiler iskelettir. Sayıların kaynağı: `results/tables/`, rapor: `report/report.pdf`.

## A. Problem ve bulanık mantık

**1. Problem nedir?** Bir öğrencinin henüz çözmediği soruyu doğru yapma olasılığını, geçmiş cevaplarından tahmin etmek. Öğrencinin becerileri doğrudan gözlenmez, cevaplardan kestirilir.

**2. Neden bulanık mantık?** Bir beceriye sahip olma durumu "var/yok" değil kısmidir (ör. kesirlerde toplama bilen ama sadeleştirmede zayıf öğrenci). Bulanık küme bunu 0-1 arası üyelik derecesiyle ifade eder; DINA ise ikilidir.

**3. Bulanık küme nedir, üyelik derecesi ne demek?** Klasik kümede eleman ya içeridedir ya dışarıda. Bulanık kümede 0 (hiç ait değil) ile 1 (tam ait) arasında derece vardır. Bizde: "beceri k'yı bilenler" kümesinde öğrencinin derecesi a_ik.

**4. Fuzzification, rule base, inference, defuzzification sisteminizde nerede?**
- Fuzzification: θ'dan sigmoid ile a_ik (`src/memberships.py`).
- Kural tabanı: Q matrisi. Objektif soru: "beceri 1 VE beceri 2 ...", subjektif soru: "beceri 1 VEYA beceri 2 ...".
- Çıkarım: t-norm (VE) / t-conorm (VEYA) ile η (`src/fuzzy_ops.py`, `src/mfcd.py`).
- Durulaştırma: Klasik centroid yok. P(doğru) = (1-s)η + g(1-η) bir olasılık; 0.5 eşiği ile "doğru/yanlış" kararı bunun karşılığıdır. Bunu dürüstçe söyleyin.

**5. Mamdani mı Sugeno mu?** İkisi de değil. Kural tabanlı çıkarım var ama kurallar Q matrisinden gelen sabit VE/VEYA yapısıdır, çıkış bir olasılıktır. Type-2, ANFIS, Fuzzy C-Means kullanmadık.

**6. t-norm ve t-conorm nedir?** Bulanık VE ve VEYA. Gödel: min / max. Çarpım: a·b / a+b−ab. Łukasiewicz: max(0,a+b−1) / min(1,a+b). Hepsi klasik VE/VEYA'yı 0 ve 1'de aynen verir, aralarda farklıdır.

**7. Neden objektifte VE, subjektifte VEYA?** Objektif soruda (çoktan seçmeli hesap) gereken tüm beceriler lazım. Subjektif soruda (yazma, çeviri) bir beceri diğerini telafi edebilir.

## B. Üyelik fonksiyonu ve parametreler

**8. Neden sigmoid?** Yetenek arttıkça yeterlilik artar (monoton), 0-1 arasında kalır, IRT'deki lojistik modelle uyumlu. Makalenin Eş. 1'i. Üçgen/gauss gibi alternatifleri denemedik (sınırlılık).

**9. α, β, θ ne işe yarar? 1.7 neden?** θ: öğrencinin yeteneği. β: becerinin zorluğu (θ=β'da a=0.5). α: ayırt edicilik (büyükse geçiş dikleşir; küçük α'da yetenek farkı derece farkına az yansır). 1.7 sigmoidi normal dağılım eğrisine yaklaştıran klasik IRT sabitidir.

**10. Slip ve guess nedir?** Slip: biliyor ama yanlış yapma (dikkatsizlik). Guess: bilmiyor ama doğru tahmin. P = (1-s)η + g(1-η) → P her zaman [g, 1-s] aralığında, log(0) olmaz.

**11. Bilgi açığı nedir, μ ne?** μ_jk becerinin o soru için önemi (0-1). Objektifte ã = 1 − μ(1 − a): önemli bir beceride eksiklik daha çok cezalanır. Gödel ile birleşince makalenin Eş. 8-9'u olur. İkili Q'da (μ=Q) tam olarak min'e iner (testle doğruladık), bu yüzden FrcSub'da bilgi açığı ek bilgi taşımaz.

**12. Zaman ve süre ağırlığı?** Zaman: ω=(T_m−Δt)/T_m, eski kayıt az güvenilir. Süre: hızlı ve doğru cevap daha güvenilir. İkisini de gözlem ağırlığı olarak uyguladık; bu bizim yorumumuz (makale belirsiz). Makalenin Eş. 3 yazımı yeterliliği çarpmak; onu da ayrı ayar olarak denedik.

## C. Algoritmalar

**13. Metropolis-Hastings nasıl çalışır?** Parametreler için rastgele bir değişiklik önerilir; olabilirlik×önsel artıyorsa kabul, azalıyorsa olasılıkla kabul edilir. Binlerce adım sonra örnekler posterior dağılımı temsil eder; tahmin posterior ortalamadır. Neden MAP/EM değil: K=25 beceri ve gizli yapı yüzünden EM pratik değil; MH basit ve hazır gradyan gerektirmiyor.

**14. Önseller ve adım boyları?** θ,β~N(0,1); log α~N(0,0.5²); s,g~U(0,0.4); μ~U(0,1]. Bizim seçimimiz. Adım boyları kabul oranı 0.25-0.55 olacak şekilde elle ayarlandı. Önsel duyarlılığını test etmedik (sınırlılık).

**15. R-hat nedir? Neden 3 başlangıç?** R-hat: farklı başlangıçlı zincirlerin birbirine ne kadar uyduğu (1.0 ≈ yakınsamış; >1.1 şüpheli). FrcSub'da MFCD-core çok modlu: tek zincir bazen kötü çukura düşüyor (120 koşunun 44'ünde başlangıçlar arası fark >30 log-olabilirlik). Eğitim olabilirliği en yüksek zinciri seçiyoruz, test'e bakmıyoruz (sızıntı yok).

**16. Baseline'lar nedir?** IRT-2PL (MAP, L-BFGS), DINA (Gibbs; ikili beceri + slip/guess), FuzzyCDF-orig (μ=Q, ağırlıksız MH). Hepsi aynı bölmede, aynı seed'lerle.

**17. Metrikler?** PRE/Recall/F1 (doğru = pozitif, eşik 0.5), MAE, RMSE, AUC (eşikten bağımsız sıralama başarısı), kalibrasyon (ortalama tahmin − gerçek oran).

## D. Deneyler ve sonuçların nedenleri

**18. Makaleyi yeniden üretebildiniz mi?** Kısmen. Eğitim oranıyla iyileşme ve DINA<FuzzyCDF (F1) uyuştu. MFCD-core, FuzzyCDF-orig'i geçmedi (F1 ≈ −0.003 ... −0.005, RMSE aynı); DINA'nın PRE'si bizde en yüksek, makalede en düşük. Nedenler: FrcSub'da zaman/süre/öznel soru yok; Q ikili; makalenin sıçraması MAKT'tan olabilir; ağırlık uygulaması belirsiz. Hiçbirini ayrıca test etmedik.

**19. Makale yanlış mı?** Hayır dedik, demiyoruz. Biz makalenin verisini, MAKT'ını ve kesin ağırlık formülünü bilmiyoruz. "Bizim koşullarımızda yeniden üretilemedi" diyoruz.

**20. K1 neydi, ne buldunuz?** Operatör ailelerini (Gödel/çarpım/Łukasiewicz) aynı koşullarda karşılaştırdık. FrcSub'da fark yok (%60 F1 0.849/0.848/0.847). Gödel verisinde Gödel kazandı ama veriyi Gödel ile ürettiğimiz için beklenen. Çarpım verisinde bu sıralama çıkmadı; farklar çok küçük. Yani sıralama üretece bağlı.

**21. K2'nin ana bulgusu?** "+süre" F1'i 0.618→0.671 çıkardı ama negatif kontrolde (süre cevaptan bağımsız) de arttı (0.678→0.709). Neden: ağırlık yalnızca doğru cevaplara bonus veriyor, model "doğru" demeye kayıyor (ort. tahmin 0.58, gerçek 0.48); doğru sınıf pozitif olduğu için F1 artıyor. AUC (sıralama) hiçbir faktörle artmadı.

**22. Negatif kontrol nedir, neden önemli?** Faktörün dayandığı bilgiyi bilerek anlamsız yaptığımız veri. Orada da "iyileşme" görüyorsak iyileşme bilgiden değil yöntemin yan etkisinden gelir.

**23. Sentetik veri döngüsel değil mi?** Evet; faktörlerle ve Gödel ile ürettik. Bu yüzden sonuçlar "geri kazanılabilir mi" sorusuna cevap verir, gerçek yararı kanıtlamaz. Çarpım verisi kontrolü ve negatif kontrol bu riski azaltır ama gerçek veri yerine geçmez.

**24. Süre karşılaştırması?** E4: tek işlem, tek zincir. FrcSub: IRT 0.03 sn, DINA 1.33, FuzzyCDF-orig 6.21, MFCD-core 7.11. Sentetik: 0.11 / 5.86 / 54.35 / 64.82. Paralel koşulardaki süre sütunları kıyaslanamaz (10 işlem birbirini yavaşlattı).

**25. Parametreleri yorumlayabilir misiniz (ör. hangi becerinin μ'si yüksek)?** Hayır. Sentetikte tahmin yakınsıyor ama bireysel α, β, μ, s, g zayıf belirlenmiş (R-hat>1.1). Yalnızca tahminleri kullanıyoruz.

## E. Kod ve süreç

**26. Kodun ana parçaları?** `src/memberships.py` (Eş. 1), `src/fuzzy_ops.py` (t-norm/t-conorm), `src/mfcd.py` (η ve olasılık), `src/mcmc.py` (MH), `src/baselines.py` (IRT, DINA), `src/metrics.py`, `src/experiment.py` (koşucu), `src/data_synthetic.py`, `demo/canli_demo.py`.

**27. Yazılımın doğruluğunu nasıl kontrol ettiniz?** 69 birim testi: elle hesaplı t-norm/t-conorm değerleri, bilgi açığı min'e indirgeme (rastgele girdilerle, literal formül döngüsüne karşı), Łukasiewicz uç değerleri, gradyan kontrolü, metrikler, veri boyutları; aynı seed'le yeniden koşu birebir aynı sonuç; temiz klon testi.

**28. Üretken yapay zekâ kullandınız mı?** Evet: Claude Code ile kod, hata ayıklama, deney tasarımı tartışması, kavram açıklaması, kaynak doğrulama yardımı. Rapordaki bölümde yazılı (Generative AI Usage Disclosure). Kod ve sonuçlar sizin tarafınızdan anlaşılmış olmalı: demo ve bu soruları kendiniz cevaplayabilmelisiniz.

**29. Resmi makale kodu kullandınız mı?** Hayır; makale kodu kullanılmadı. Tüm kod bu projede yazıldı. Veri: FrcSub (Wu vd., IJCAI 2015) indirilir, repoya konmaz.

## F. Sınırlılıklar (kısa liste, ezberleyin)
MAKT yok; makale verisi kapalı (yalnızca FrcSub: ikili, objektif, zaman/süre yok); ağırlık uygulaması ve Eş. 7 normalizasyonu bizim yorumumuz; sentetik veri döngüsel; K1/K2 sentetik kısmı yalnızca %60 oranında; MFCD-core FrcSub'da çok modlu; bireysel parametreler yorumlanamaz; 10 seed, çoklu karşılaştırma düzeltmesi yok; önsel duyarlılığı test edilmedi.

## G. "Şunu canlı değiştirin" tipi istekler
- Bir öğrencinin yeteneğini değiştir: `python3 -m demo.canli_demo --ogrenci 12 --soru 5 --theta 2.0` (θ yükselince a→1, η→1, P→1−s).
- Operatörleri yan yana gör: `python3 -m demo.canli_demo --ogrenci 12 --soru 3 --theta 0.3` (4 becerili soru; aynı üyeliklerle Gödel ≥ çarpım ≥ Łukasiewicz satırı çıkar; Gödel=min en büyük, Łukasiewicz en küçük). Tek becerili sorularda (5, 7, 8) üç operatör aynı η'yı verir.
- `--operator product` modeli baştan o operatörle eğitir; üyelikler de değişir, bu yüzden "saf" fark için yukarıdaki satıra bakın.
- Beklenmedik sonuç çıkarsa (ör. yanlış tahmin): "model gerçek cevabı bilmiyor, olasılık verir; 0.43 kararı YANLIŞ yapar ama olasılık 0.5'e yakın, belirsizlik yüksek" diye açıklayın.
