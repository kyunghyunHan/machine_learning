#지도 학습알고리즘은 분류와 회귀
#회귀는 예측하는 문제
#최근접 = 근처3개중에 2개가 그거면 
#샘플은 임의의 수치
#평균
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
#산점도
from sklearn.model_selection import train_test_split

plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#훈련세트와 테스트 세트 나누기
train_input,test_input,train_target,test_target= train_test_split(
    perch_length,perch_weight, random_state= 42
)


test_array = np.array([1,2,3,4])
print(test_array.shape)
#바꾸려난 베열의 크기를 지정가능
test_array = test_array.reshape(2,2)
print(test_array.shape)

train_input= train_input.reshape(-1,1)
test_input= test_input.reshape(-1,1)
print(train_input.shape,test_input.shape)


from sklearn.neighbors import KNeighborsRegressor

knr=KNeighborsRegressor()

knr.fit(train_input,train_target)

print(knr.score(test_input,test_target))
#회귀의 경우 이점수를 결정계수.간단히 R2
# (타킷-예측)²의합/(타킷-평균)²의합

from sklearn.metrics import mean_absolute_error

#테스트세트에 대한 예측
test_prdiction=knr.predict(test_input)

# 테스트 세트에 대한 평균 절댓값 오차계산
mae= mean_absolute_error(test_target,test_prdiction)
print(mae)

##결과에서 예측이 평균 19g정도 다르다는것을 알수있다
print(knr.score(train_input,train_target))

#과대적합:훈련세트에서 점수가 좋앗는데 테스트가 너무 나쁠경우

#과소적합:훈련세트보다 테스트세트의 점수가 높거나 두점수 모두 낮을경우 
#이웃의 개수를 3으로변경
knr.n_neighbors= 3
#모델훈련
knr.fit(train_input,train_target)
print(knr.score(train_input,train_target))

print(knr.score(test_input,test_target))

#회귀는 임의의 수치를 예측하는 문제
#결정계수:대표적인 회귀 문제의 성능 측정 도구 1에가까울수록 좋고 0에 가깝다면 성능이 나쁜모델

#mean_absolute_error():회귀모델의 평균 절댓값 오차를계산
#첫번째 매개변수는 타겟,두번째는 예측ㄱ밧