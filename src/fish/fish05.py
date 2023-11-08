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

print(knr,predict([[50]]))

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

#선형회귀:특성이 하나인 경우

from kelearn.linear_model import LinearRegression
lr = LinearRegression()

#선형회귀 모델 훈련
lr.fit(train_input,train_target)

print(lr.predict([[50]]))

print(lr.coef_,lr.intercept_)

#훈련세트의 산점도

plt.scatter(train_input,train_target)

plt.plot([15,50],[15*lr.intercept_,50*lr.coef_+lr.intercept_])

#50cm
plt.scatter(50,1241.8,marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(lr.score(train_input,train_target))#훈련세트
print(lr.score(test_input,test_target))#테스트세트

##다항회귀
train_poly = np.colum_stack((train_input **2,train_input))
test_poly = np.colum_stack((test_input **2,test_input))

print (train_poly.shape,test_poly.shape)

lr= LinearRegression()
lr.fit(train_poly,train_target)

print(lr.predict([[50**2,50]]))

print(lr.coef_,lr.intercept_)

#구간별 직전을 그리기 위해 15에서 49까지 정수배열
point= np.arange(15,50)
#훈련세트의 산점도
plt.scatter(train_input,train_target)