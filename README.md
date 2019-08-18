# Detection-and-Recognition-of-Highway-Traffic-Panels

本論文以深度學習作為基礎，對國道上的交通路牌進行偵測與辨識，讓駕駛者可以透過此系統得知前方路牌資訊。偵測方面，使用深度可分離卷積結構(Depthwise Separable Convolution)，與單階段的物件偵測模型，在複雜的背景中偵測路牌。辨識方面，利用前饋式類神經網路(FeedForward Neural Network，FFNN)來辨識路牌資訊。實驗結果表明路牌偵測模型的精確率(Precision)以及召回率(Recall)有不錯的表現且不同的特徵擷取架構會影響移動式裝置上的測試時間而路牌辨識模型的準確率(Accuracy)達到99%的表現。

## 系統架構
![系統架構](https://i.imgur.com/WpWA47f.png)
### 路牌偵測網路架構
![路牌偵測網路架構](https://i.imgur.com/ja541Cm.png)
### 路牌辨識網路架構
![路牌辨識網路架構](https://i.imgur.com/rErCq5Q.png)


### 實驗結果
![實驗結果](https://i.imgur.com/1xT1yUC.png)
