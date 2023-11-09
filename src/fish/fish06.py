
#특성공학
#선형회귀는 특성이 많을수록 좋음

#다중회귀
#특성이 2개면 선형회귀는 평면을 학습
#특성이 2개면 타깃값과 함계 3차원 공간형성
#타깃 = a x 특성1 x 특성2+절편은 평면
#특성이 많은 고차원에서는 선형회귀가 매우 복잡한 모델을 표현할수 있다
#농어의 길이 + 농어의 높이 +두께
# 농어길이  * 농어 높이 새로운 특성으로 재탄생->특성공학
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
df = pd.read_csv('https://bit.ly/perch_csv_data')
prech_full = df.to_numpy()
print(prech_full)


import numpy as np

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

#훈련세트와 테스트 세트 나누기
from sklearn.model_selection import train_test_split

train_input,test_input,train_target,test_target= train_test_split(
    prech_full,perch_weight, random_state= 42
)

#사이킷런 변환기

from sklearn.preprocessing import PolynomialFeatures
#변환기는 타깃데이터가 필요없음
#PolynomialFeatures는 각 특성을  제곱한 항을 추가하고 특성끼리 서로 곱한 항을 추가
# 무게 = a * 길이 +b *높이 +c * 두께 +d *1
#선형방정식의 절편은 항상 값이 1인 특성과 곱해지는 계수
poly=PolynomialFeatures(include_bias=False)
poly.fit([[2,3]])
print(poly.transform([[2,3]]))

poly=PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)

#각각 특성이 어떤입력의 조합으로 이루어졋는지 확인가능
poly.get_feature_names_out()

test_poly= poly.transform(test_input)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#선형회귀 모델 훈련
lr.fit(train_poly,train_target)
print(lr.score(train_poly,train_target))
print(lr.score(test_poly,test_target))

#특성 추가

poly=PolynomialFeatures(degree=5,include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly= poly.transform(test_input)
#특성 55개
print(train_poly.shape)
lr.fit(train_poly,train_target)
print(lr.score(train_poly,train_target))
#선형모델은특성의 개수를 늘리면 훈련에대해 완벽하지만 과대적합대므로 테스트 세트에서는 형편없는 점수
print(lr.score(test_poly,test_target))#음수가나옴

#규제
#머신모델이 훈련세트를 과도하게 학습하지 못하도록 훼방

from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

#선형회귀에 규제를 추가한 모델을 릿지와 라쏘


from sklearn.linear_model import Ridge
ridge= Ridge()
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))


import matplotlib.pyplot as plt
train_score=[]
test_score=[]

alpha_list=[0.001,0.01,0.1,1,10,100]

for alpha in alpha_list:
    #릿지 모델을 만듭니다
    ridge= Ridge(alpha=alpha)
    #릿지모델을 훈련
    ridge.fit(train_scaled,train_target)
    #훈련 점수와 테스트 점수를 저장
    train_score.append(ridge.score(train_scaled,train_target))
    test_score.append(ridge.score(test_scaled,test_target))

plt.plot(np.log10(alpha_list),train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

ridge= Ridge(alpha=0.1)
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))

#라쏘회귀

from sklearn.linear_model import Lasso
lasso= Lasso()
lasso.fit(train_scaled,train_target)
print(lasso.score(train_scaled,train_target))

print(lasso.score(test_scaled,test_target))

train_score=[]
test_score=[]

alpha_list=[0.001,0.01,0.1,1,10,100]

for alpha in alpha_list:
    #릿지 모델을 만듭니다
    lasso= Lasso(alpha=alpha,max_iter=10000)
    #릿지모델을 훈련
    lasso.fit(train_scaled,train_target)
    #훈련 점수와 테스트 점수를 저장
    train_score.append(lasso.score(train_scaled,train_target))
    test_score.append(lasso.score(test_scaled,test_target))

plt.plot(np.log10(alpha_list),train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()