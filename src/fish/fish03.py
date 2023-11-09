# 데이터 전처리
#생성의 길이
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
#생성의 무게
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

import numpy as np
# 전달받은 리스트를 일렬로   새운다음 나란히 연결
np.column_stack(([1,2,3],[4,5,6]))

fish_data = np.column_stack((fish_length,fish_weight))
#1로 채운 5개의 원소를가진 배열
print(np.ones(5))

fish_target = np.concatenate((np.ones(35),np.zeros(14)))

from sklearn.model_selection import train_test_split
#기본적으로 25%를 테스트 세트로 뗴어냅니다
#훈련데이터와 테스트데이터 나누기
#섞어야 골로구 데이터가 퍼짐
train_input,test_input,train_target,test_target= train_test_split(
    fish_data,fish_target,stratify = fish_target, random_state= 42
)


from sklearn.neighbors import KNeighborsClassifier

kn=KNeighborsClassifier()
kn.fit(train_input,train_target)
 
kn.score(test_input,test_target)
#이상하게나옴 빙어처럼나옴
print(kn.predict([[25,150]]))

import matplotlib.pyplot as plt

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25,10,marker='^')#marker매개변수는 모양을 지정합니다
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25,10,marker='^')#marker매개변수는 모양을 지정합니다
plt.scatter(train_input[indexes,0],train_input[indexes,1],marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


print(train_input[indexes])

print(train_target[indexes])

print(distances)
#x축은 좁고 y축은 넓기 떄문에
#거리기반이기 떄문에 일정한기준으로 맞춰주어야 함

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25,10,marker='^')#marker매개변수는 모양을 지정합니다
plt.scatter(train_input[indexes,0],train_input[indexes,1],marker='D')
plt.xlim((0,1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#전처리 방법중 하나는 표준점수
#표준점수와 표준편차
#분산은 데이터에서 평균을 뺀값을 모두 제곱한 다음 평균을 내어 구합니다
# 표준편차는 분산의 제곱근으로 데이터가 분산된 정도를 나타냅니다
#표준 점수는 각 데이터가 원점에서 표준편차만큼 떨어져 있는지를 나타내는 값입니다.
#평군 계산
mean = np.mean(train_input,axis=0)#axis=0:행을 따라 각 열의 통계값 산
#표준 편차 계산
std= np.std(train_input,axis=0)
#브로드캐스팅
# 각  원본데이터에서 평균을 빼고 표준편차로 나누어 표준 점수로 반환
train_scaled= (train_input - mean)/std


plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25,10,marker='^')#marker매개변수는 모양을 지정합니다
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
#x축과 y축의 범위가 바뀌었기 때문에 비슷한 범위를 차지함
new= ([25,150]-mean)/std
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(new[0],new[1],marker='^')#marker매개변수는 모양을 지정합니다
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
#후련을 마치고 테스트세트로 평가할떄는 테스트세트도 훈련세트의 평균과 표준편차로 변환해야한다
kn.fit(train_scaled,train_target)

test_scaled= (test_input - mean)/std

kn.score(test_scaled,test_target)

print(kn.predict([new]))

distances,indexes= kn.kneighbors([new])
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(new[0],new[1],marker='^')#marker매개변수는 모양을 지정합니다
plt.scatter(train_scaled[indexes,0],train_scaled[indexes,1],marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

#데이터 전처리는 머신모델 훈련데이터를 주입하기 전에 가공하는 단계

#표준점수는 훈련세트의 스케일을 바꾸는 방법
#표준점수를 얻으려면 특성의평귱을 뺴고 표준편차로 
