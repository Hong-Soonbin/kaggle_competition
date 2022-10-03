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
키포인트 아이디어
1. loading 피처는 피처중 가장 강한 타겟 예측력을 가지고 있다 따라서 결측값 대체에 최종 점수가 좌우한다.
  - loading==[target==1], loading==[target==0] 두 경우 모두 정규분포임을 증명
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


2. measurements 피처들간의 상관관계를 통해 결측값 대체!
  - product code 별 measurements3-9 피처들로 measurements17의 값을 선형회귀로 대체
  
  $$ X = -\frac{[\sum^9_{i=4}C_i X_i I(C_i>=0.1)] - \frac{X_17-b}{m}}{C} $$
   
3. RobustScaler
-------------------------------------------------------------------------------------------------------------------------
회고
-------------------------------------------------------------------------------------------------------------------------
첫 캐글 도전으로 playground 대회에 참가했는데 379/1889 등을 했다.<br>
discussion탭을 통해서 처음보는 방법론을 이해하는데에 시간이 많이 들어서 내가 세운 가설을 검증하기에 시간이 부족했다.<br>
다음 대회에서는 다른 사람의 검증된, 새로운 방법론도 좋지만 연습을 위한 대회인 만큼 내가 스스로 가설을 세우고 검증하는 프로세스를 더 많이 경험해 봐야겠다.
