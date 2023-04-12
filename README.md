# iFamilySC_Sentiment_Model
## 아이패밀리 프로젝트

### Requirements
* python >= 3.7.13
* pytorch >= 1.11.0
* transformers >= 4.18.0
* scikit-learn >= 1.0.2
* numpy
* pandas
* tqdm

### Sentiment Model
* 기업의 실 서비스를 위한 프로젝트 진행
  * 목표 : 댓글을 통한 긍, 부정, 중립의 분류

### Data
* 화장품에 대한 데이터를 수집하여 진행
 * Gold Label 제작

![아이패밀리 1](https://user-images.githubusercontent.com/100681144/231500251-443dfd2a-184a-4802-b58b-aeab7fc9623e.PNG)


* Label
  * 긍정 : 2
  * 부정 : 1
  * 중립 : 0

### Model
* BERT-base
  * klue/bert-base
  * Layer = 12
  * Hidden size = 768
  * Attention heads = 12
  * Total Parameters : 110M
  
### Imbalanced Data
* 긍정 데이터가 압도적으로 많음
  * 따라서 처리방법을 강구해야 함

![아이패밀리 2](https://user-images.githubusercontent.com/100681144/231502582-ce083a88-7623-4c32-a342-4df867238681.PNG)

* RandomOverSampler 사용

![아이패밀리 3 랜덤오버 데이터증강](https://user-images.githubusercontent.com/100681144/231504237-97383a72-e677-49d5-837a-e8b8c73672c3.PNG)
![아이패밀리 3 랜덤오버](https://user-images.githubusercontent.com/100681144/231504259-781463be-c2bf-4e34-8746-e3fdebdf7194.PNG)

* EDA

![아이패밀리 4 EDA 데이터증강](https://user-images.githubusercontent.com/100681144/231504413-06db2867-f1ee-4d36-9286-5942f621977d.PNG)
![아이패밀리 4 EDA](https://user-images.githubusercontent.com/100681144/231504427-452ad9b6-50b4-4453-bb05-44b9d641102a.PNG)

* Label Smoothing

![아이패밀리 5 LS](https://user-images.githubusercontent.com/100681144/231504497-11e4c409-e74f-49d2-bf47-51de160e331b.PNG)

### 결과

* 데이증강을 통한 Label Smoothing의 적용은 성능 하락을 보이므로 적용하지 않음
