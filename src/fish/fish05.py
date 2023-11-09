#최근접 이웃의 한계
#
import numpy as np

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target= train_test_split(
    perch_length,perch_weight, random_state= 42
)

train_input= train_input.reshape(-1,1)
test_input= test_input.reshape(-1,1)


from sklearn.neighbors import KNeighborsRegressor

knr= KNeighborsRegressor(n_neighbors= 2)
#k-최근접 이웃 회귀 모델을 훈련
knr.fit(train_input,train_target)

print(knr.predict([[50]]))

#50cm의 농어를 1,033g정도로 예축

import matplotlib.pyplot as plt
#50cm의 농어의 이웃
distances,indexes = knr.kneighbors([[50]])

#훈련세트의 산점도


from sklearn.model_selection import train_test_split

plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes],marker='D')
plt.scatter(50, 1033,marker='^')

plt.xlabel('length')
plt.ylabel('weight')
plt.show()
#최근접 이웃은 평균을 구하기때문에 새로운 샘플이 범위를벗어나면 이상한 값 예측

print(knr.predict([[100]]))

distances,indexes = knr.kneighbors([[100]])

#후련세트의 산점도
plt.scatter(train_input, train_target)

plt.scatter(train_input[indexes],train_target[indexes],marker='D')

plt.scatter(100, 1033,marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#최근접 을 통해 문제를 해결하려면 가장 큰 농어가 포함되도록 훈련세트를 다시 만들어야 합니다

#선형회귀: 간단하고 성능이 뛰어남 특성이 하나인 경우 어떤 직선을 학습하는 알고리즘

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#선형회귀 모델 훈련
lr.fit(train_input,train_target)
## 50cm 농어에 대해 예측
print(lr.predict([[50]]))
#한나의 직선을 그리려면 기울기와 절편이 있어여함
# y = a * x + b
#x를 농어의 길이 
#기울기를 a
#절편을 b
#Y를 농어의 무게
# a = coef_
# b= intercept_
# 머신러닝에서는 기울기를 계수,또는 가중치라함
print(lr.coef_,lr.intercept_)

#훈련세트의 산점도

plt.scatter(train_input,train_target)
# 15에서 50까지 1차방정식 그래프 그리기
plt.plot([15,50],[15*lr.coef_+lr.intercept_,50*lr.coef_+lr.intercept_])

#50cm 농어 데이터
plt.scatter(50,1241.8,marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(lr.score(train_input,train_target))#훈련세트
print(lr.score(test_input,test_target))#테스트세트
## 그래프가 직선이면안댐 0g이하로 내려가기 떄문에
# 무게  = a * 길이 ² + b * 길이 + c
##다항회귀
#colum_stack을 이용하여 제곱한것을 나란히 붙
train_poly = np.column_stack((train_input **2,train_input))
test_poly = np.column_stack((test_input **2,test_input))

print (train_poly.shape,test_poly.shape)

lr= LinearRegression()
lr.fit(train_poly,train_target)
#제곱한 것을 같이 넣어주어야 함
print(lr.predict([[50**2,50]]))

print(lr.coef_,lr.intercept_)
#다항식을 사용한 선형회귀를 다항회귀
#구간별 직전을 그리기 위해 15에서 49까지 정수배열
point= np.arange(15,50)
#훈련세트의 산점도
plt.scatter(train_input,train_target)

#15에서 49까지 2차 방정식 그리기
plt.plot(point,1.01*point**2 - 21.6*point + 116.05)

# 50cm농어 데이터
plt.scatter(50,1574,marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(lr.score(train_poly,train_target))
print(lr.score(test_poly,test_target))