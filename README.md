# Churn Analizi Projesi

Bu proje, Flask web framework'ü kullanılarak geliştirilmiş bir müşteri kaybı (churn) analiz ve tahmin uygulamasıdır. Proje, veri ön işleme, keşifsel veri analizi (EDA), makine öğrenimi modelleri ve Flask tabanlı bir web arayüzü içermektedir.


## İçindekiler

1. [Giriş](#giriş)
2. [Veri Seti](#veri-seti)
3. [Veri Ön İşleme](#veri-ön-işleme)
4. [Keşifsel Veri Analizi (EDA)](#keşifsel-veri-analizi-eda)
5. [Özellik Mühendisliği](#özellik-mühendisliği)
6. [Modelleme](#modelleme)
7. [Model Değerlendirme](#model-değerlendirme)
8. [Sonuç ve Değerlendirme](#sonuç-ve-değerlendirme)
9. [Kullanılan Kütüphaneler](#kullanılan-kütüphaneler)


## Giriş

Müşteri kaybı (churn), bir şirketin mevcut müşterilerinin belirli bir süre içinde hizmet veya ürünü kullanmayı bırakması anlamına gelir. Bu proje, churn tahmin modelleri geliştirerek müşteri kaybını önlemek ve müşteri sadakatini artırmak amacıyla yapılmıştır.

## Veri Seti

Bu projede kullanılan veri seti, [Veri Kaynağı](#) bağlantısından indirilebilir. Veri seti, müşterilere ait demografik bilgiler, kullanım alışkanlıkları ve abonelik detayları gibi bilgileri içermektedir.

## Veri Ön İşleme

Veri seti üzerinde yapılan ön işlemler şunlardır:
- Eksik değerlerin impute edilmesi
- Kategorik değişkenlerin encode edilmesi
- Aykırı değerlerin tespiti ve işlenmesi

## Keşifsel Veri Analizi (EDA)

Veri setinin genel özelliklerinin anlaşılması ve önemli değişkenlerin tespiti için çeşitli görselleştirmeler ve istatistiksel analizler yapılmıştır. Bu analizlerde:
- Churn ve churn olmayan müşterilerin dağılımı
- Demografik değişkenlerin churn ile ilişkisi
- Kullanım alışkanlıklarının analizi

## Özellik Mühendisliği

Model performansını artırmak amacıyla yeni özellikler oluşturulmuş ve mevcut özellikler dönüştürülmüştür. Bu süreçte:
- Kullanım süreleri ve sıklıklarına dayalı yeni değişkenler oluşturulmuştur
- Kategorik değişkenler dummy değişkenlere dönüştürülmüştür

## Modelleme

Müşteri kaybını tahmin etmek için çeşitli makine öğrenimi algoritmaları kullanılmıştır:
- Lojistik Regresyon
- Karar Ağaçları
- Random Forest
- XGBoost

## Model Değerlendirme

Modellerin performansını değerlendirmek için aşağıdaki metrikler kullanılmıştır:
- Doğruluk (Accuracy)
- Hata Matrisi (Confusion Matrix)

## Sonuç ve Değerlendirme

Modellerin sonuçları karşılaştırılmış ve en iyi performansı gösteren model seçilmiştir. Ayrıca, modelin iş kararlarına etkisi değerlendirilmiştir.

## Kullanılan Kütüphaneler

Bu projede aşağıdaki Python kütüphaneleri kullanılmıştır:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

