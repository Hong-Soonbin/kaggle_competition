상품의 불량 여부를 이진분류하는 문제 (47300,26)
-------------------------------------------------------------------------------------------------------------------------
캐글 진행과정
1. 데이터셋 타겟 비율 체크
2. 결측치 체크
3. 피처 요약표 사용
4. 데이터 타입별 적절한 시각화 
  - 타겟 비율
  - 피처 별 타겟 분포 sns.histplot(dtypes in ['int','float']), sns.countplot(dtypes == 'object') hue=target
  - 피처 별 타겟 비율, 신뢰구간 sns.barplot
  - sns.boxplot 이상치 확인
  - tarin, test 데이터셋간의 분포 비교, target을 sns.scatter()로 ax.twinx()에 표시
  - 상관관계 시각화 sns.heatmap
5. 결측치
  - simple impute 평균, 중앙값, 최빈값 등등으로 대체    #(성능하락)
  - HuberRegressor model(이상치에 둔감함 선형모델) 상관관계가 있는 피처를 사용해서 결측치를 예측
    https://towardsdatascience.com/regression-in-the-face-of-messy-outliers-try-huber-regressor-3a54ddc12516
  - 결측치 값 자체가 타겟예측에 영향을 끼치는치 확인 후 결측치 자체를 피처로 생성
6. Encoding
  - StandardScaler
  - WoEEncoder
    - 이진 피처를 0,1이 아닌 WOE = In(% of non-events ➗ % of events) 다음의 수식을 통해서 조금더 의미있는 값으로 대체
7. 학습방법
  - GroupKFold사용
    EDA중 product_code 마다 나올 수 있는 attribute의 조합이 정해져 있기 때문에 GroypKFold사용
  - LogisticRegression 모델 l2 = 0.0001
8. score
  - public = 0.59013, private = 0.5943

-------------------------------------------------------------------------------------------------------------------------
키 포인트 아이디어
1. loading 피처는 피처중 가장 강한 타겟 예측력을 가지고 있다 따라서 결측값 대체에 최종 점수가 좌우한다.
  - loading[target==1], loading[target==0] 두 경우 모두 정규분포를 따른다면 각각을 정규분포 식에 넣어 '서로 같다' 라는 수식을 새워 근의공식을 통해 값을 도출
   $$(\ln(loading)|failure = i) ∼ N(\mu_i, \sigma_i^2),\quad i = 0,1 $$
   두 분포 모두 정규분포를 따르므로
   
   $$\frac{1}{\sigma_0\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{\ln(x)-\mu_0}{\sigma_0})^2} = \frac{1}{\sigma_1\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{\ln(x)-\mu_1}{\sigma_1})^2}  $$
   
   이 수식이 성립하게 된다.   
   
   $$ \ln(x) = \ln(\frac{\sigma_1}{\sigma_0}) = -\frac{1}{2}(\frac{\ln(x)-\mu_1}{\sigma_1})^2 +\frac{1}{2}(\frac{\ln(x)-\mu_0}{\sigma_0})^2 $$
   
   전개
   
   $$ = \frac{(\sigma_1^2 - \sigma_0^2)(\ln(x))^2 + (2\sigma_0^2\mu_1 - 2\sigma_1^2\mu_0)\ln(x) + \mu_0^2\sigma_1^2 - \mu_1^2\sigma_0^2}{2\sigma_0^2\sigma_1^2} $$
   
   근의 공식 사용
   
   $$ x = 127.23... $$
   
   train, test의 loading 피처의 결측값을 127.23으로 대체!   

<br>

2. measurements 피처들간의 상관관계를 통해 결측값 대체!
  - product code 별 measurements3-9 피처들로 measurements17의 값을 HuberRegressor model로 대체
    train을 (measurement_n==결측치없음 & measurement_17==결측치없음)으로 두고   
    test를  (measurement_n==결측치없음 & measurement_17==결측치있음)으로 두고 학습시켰다.
    ```python
    model = HuberRegressor(epsilon=1.9)
    model.fit(tmp_train[column], tmp_train['measurement_17'])
    data.loc[(data.product_code==code)&(data[column].isnull().sum(axis=1)==0)&(data['measurement_17'].isnull()), 'measurement_17'] = model.predict(tmp_test[column])
    ```
    
   <br>
   
3. measurements3, measurements5 결측값을 새로운 피처로 생성   
  -가설: 결측값의 유무가 타겟값의 확률에 영향을 끼치는가?
  -검증
  ```python
  # 결측값인데 target==1인 경우의 수 / 결측값의 수
  df[df["measurements3"].isna()]["target"].sum() / df[df["measurements3"].isna()].sum
  # 그냥 target==1인 경우의 수 / 모든 경우의 수    == 0.212608
  df["measurements3"]["target"].sum() / df["measurements3"].sum() 
  ```
  위의 값과 아래의 값의 각각의 확률분포가 같다면 타겟예측에 영향력이 없는 경우이고   
  각각의 확률분포가 다르다면 타겟예측에 영향력이 있다고 판단할 수 있다.   
  zscore를 통해 비교해보자
  ```python
  total = train[f].isna().sum()
  fail = train[train[f].isna()].failure.sum()
  z = (fail / total - 0.212608) / (np.sqrt(0.212608 * (1-0.212608)) / np.sqrt(total))
  ```
  zscore : 지금 값이 얼마나 흔한가에 대한 지표 (결측값일때 타겟값일 확률 - 그냥 타겟값의 확률) / 표준편차(이진분류라 베르누이분포의 표준편차=np(q-p)) / n^0.5   
  measurement_3 zscore = -2.50, measurement_5 zscore =  2.66    
  - 해석
    z값이 작을수록 값이 가진 의미는 "결측치일때 타겟값이 1인 경우는 결측치 상관없이 무작위로 뽑을때의 확률과 비슷해, 흔해 라는 의미이고"   
    z값이 큰 경우는 "결측치일때 타겟값이 1인 경우는 결측치 상관없이 무작위로 뽑을때의 확률과는 다른 분포를 보이고 있어" 라는 의미이다.
    우리가 구한 zscore는 -2.5, 2.66이고 결측치 상관없이 무작위로 뽑을때 약 2% 미만의 확률로 우연히 결측치일때 타겟값이 1인 분포가 나온다고 말할 수 있다.(1종오류 유의)
    따라서 유의수준 5%미만으로 둘은 다른 분포를 가지고있고 결측치는 그 자체로 타겟예측에 영향력이 있다고 말할수있다.
  
    z score = (예측값 - 평균) / 표준편차
    하지만 여러 샘플 이 있고 해당 샘플 평균의 표준 편차( 표준 오차 )를 설명하려면 다음 z 점수 공식을 사용합니다.
    z score = (예측값 - 평균) / (표준편차/n**0.5)
    키190 평균키170 표준편차3.5 키가 정규분포를 따를 때평균키가 190인 50명의 표본을 찾을 확률
    (190 - 170) / (3.5 / 50^0.5)
    <br>
-------------------------------------------------------------------------------------------------------------------------
회고
-------------------------------------------------------------------------------------------------------------------------
첫 캐글 도전으로 playground 대회에 참가했는데 379/1889 등을 했다.<br>
통계적 기법을 직접 가설검증을 통해 구현하느라 수학공부의 필요성을 더 느끼게 되었다.<br>
discussion탭을 통해서 처음보는 방법론을 이해하는데에 시간이 많이 들어서 내가 세운 가설을 검증하기에 시간이 부족했다.<br>
다음 대회에서는 다른 사람의 검증된, 새로운 방법론도 좋지만 연습을 위한 대회인 만큼 내가 스스로 가설을 세우고 검증하는 프로세스를 더 많이 경험해 봐야겠다.
