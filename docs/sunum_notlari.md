# Sunum konuşma notları (12 slayt, yaklaşık 15-17 dakika)

Kendi cümlelerinizle söyleyin; aşağıdakiler iskelet. Süre: slayt başına ~1-1,5 dk, demo ~3 dk.

**1. Başlık (0,5 dk).** "Zhang ve Wang'ın bulanık bilişsel tanı makalesini inceledim. Bulanık çekirdeğini sıfırdan yazdım ve makalenin eklediği üç şeyin gerçekten fayda sağlayıp sağlamadığını kontrollü deneylerle sınadım."

**2. Problem (1 dk).** Soru: bir öğrenci henüz görmediği soruyu doğru yapar mı? Beceriler gözlenmez, sadece doğru/yanlış var. Klasik modelde beceri var/yok; gerçekte kısmi. Bulanık mantık bunu 0-1 arası bir derece yapar. Kullanım: zayıf becerisine uygun soru önermek.

**3. Literatür ve makale (1,5 dk).** Üç hat: klasik (DINA), bulanık (FuzzyCDF), derin (DKT, DKVMN, NeuralCDM). Makale üç bulanık ekleme yapıyor: zaman (eski cevap az güvenilir), süre (hızlı doğru cevap güvenilir), bilgi açığı (beceri soru için ne kadar önemli), ayrıca bir derin ağ (MAKT). Eleştiri: veri kapalı, ağırlıkların uygulanışı belirsiz, eklemelerin payı derin ağdan ayrıştırılmamış. Benim sorum: bulanık eklemeler tek başına fayda sağlıyor mu?

**4. Bulanık küme (1,5 dk).** Klasik küme: içeride/dışarıda. Bulanık küme: 0-1 arası derece. Grafik: yetenek arttıkça beceri derecesi artar. β zorluk (θ=β'da 0.5), α eğimin dikliği. Bu adıma fuzzification denir.

**5. Soru birden fazla beceri istiyorsa (1,5 dk).** Bir soru birkaç beceri istiyorsa dereceler birleşir. Örnek: dört beceri 0.94, 0.76, 0.66, 0.98. Bulanık VE'nin üç türü: Gödel en küçüğü alır (0.66), çarpım çarpar (0.47), Łukasiewicz daha da sert (0.34). Kural tabanı Q matrisi: objektif soruda VE, subjektifte VEYA. Sonra slip/guess ile olasılık, 0.5 eşiğiyle karar; bu, durulaştırmanın karşılığı. (Mamdani/Sugeno değil, dürüstçe söyleyin.)

**6. Sistem ve veriler (1 dk).** Çekirdeği sıfırdan yazdım (sigmoid, üç operatör ailesi, bilgi açığı, slip/guess, Metropolis-Hastings, 69 test). Baseline: IRT-2PL, DINA, FuzzyCDF-orig. Veri: gerçek FrcSub ve sentetik; sentetikte negatif kontrol var. MAKT (dikkat ağı) yok, kod baştan yazıldı; ağırlıkların uygulanışı benim yorumum (makale belirsiz bırakmış), bunu sınırlılıklarda da söyleyeceğim. Süre tablosu: MH modelleri yavaş (FrcSub'da 6-7 sn), IRT çok hızlı.

**7. FrcSub sonucu (1,5 dk).** Uyuşan: eğitim oranıyla iyileşme, DINA < FuzzyCDF. Uyuşmayan: MFCD-core FuzzyCDF-orig'i geçmedi; DINA'nın PRE'si bizde en yüksek. Test edilemeyen: öznel soru ve zaman/süre FrcSub'da yok.

**8. MCMC (1 dk).** Sentetikte tahminler yakınsıyor (R-hat ≈ 1). FrcSub'da MFCD-core başlangıca duyarlı: 120 koşunun 44'ü. Çözüm: 3 başlangıç, eğitim olabilirliği en iyi olan; test'e bakılmaz. Bireysel α, β, μ yorumlanamaz.

**9. K1 (1,5 dk).** Üç operatörü aynı koşulda kıyasladım. Fark çok küçük (F1 ≤ 0.008). Gödel verisinde Gödel kazandı ama veriyi ben Gödel ile üretmiştim; bunun için veriyi çarpımla da ürettim: sıralama çıkmadı. Yani sonuç üretece bağlı.

**10. K2 (2 dk, en önemli).** "+süre" F1'i 0.618'den 0.671'e çıkardı, ilk bakışta iyi. Ama süre bilgisini cevaptan bağımsız yaptığım negatif kontrolde de arttı. Yani fayda süre bilgisinden gelmiyor. Neden: ağırlık yalnızca doğru cevaplara bonus veriyor, model "doğru" demeye kayıyor, doğru sınıf pozitif olduğu için F1 şişiyor. Eşikten bağımsız AUC hiç artmadı. Ders: F1'e tek başına güvenme.

**11. Sonuç ve sınırlar (1,5 dk).** Çekirdek çalışıyor; makalenin eklemelerinin yararı bizim koşullarımızda yeniden üretilemedi (makale yanlış demiyorum: veri, MAKT ve kesin formül bende yok). Sınırlar: MAKT yok, ağırlık yorumu benim, sentetik veri döngüsel ve yalnızca %60 oranı, FrcSub'da çok modluluk, 10 seed. Gelecek: süreyi cevabın olasılığına bağlayan model, öznel sorulu gerçek veri.

**12. Demo (3 dk).** (1) Bir öğrenci ve soru: üyelik derecelerini, VE'yi, olasılığı, kararı göster. (2) `--theta 2.0`: yetenek yükselince derece 1'e, olasılık ~1'e gider. (3) Dört becerili soruda üç operatörün η'sını yan yana göster: Gödel ≥ çarpım ≥ Łukasiewicz. Sayıların deney sonucu değil, kısa bir koşu olduğunu söyleyin.

**Olası sorular:** docs/savunma_hazirlik.md
