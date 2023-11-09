# 머신러닝은 크게 지도학습과비지도학습

##지도학습:데이터와 정답을 입력과 타겟이라하고 합쳐서 훈련데이터 합니다
##지도학습은 정답이 있으니 알고리즘이 정답을 맞히는 것을 학습

##비지도 학습은 데이터를 잘 파악하거나 변형하는데 도움을 줌
## 연습문제와 시험문제가 달라야 하듯이 훈련데이터와 평가에 사용될 데이터가 각각 달라야함

##평가에 사용되는 데이터를 테스트세트
## 훈련에 사용되는 데이터를 훈련세트


# 훈련 셋, 테스트 셋
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

## 하나의 생성 데이터를 샘플이라 부릅니다.
## 35개를 데이터세트 14개를 테스트세트
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()

print(fish_data[4])#29.0,430.0
#파이썬 리스트는 인덱스 외에도 슬라이싱 이는 특별한 연산자
print(fish_data[0:5])

print(fish_data[:5])

print(fish_data[44:])
# 훈련세트로 입력값 중 0부터 34번째 인덱스 까지사용
train_input = fish_data[:35]
# 훈련세트로 타깃값 중 0부터 34번째 인덱스 까지사용

train_target = fish_target[:35]
# 테스트세트로 입력값 중 35부터 마지막 인덱스 까지사용

test_input = fish_data[35:]
# 테스트세트로 타깃값 중 35부터 마지막 인덱스 까지사용

test_target = fish_target[35:]
#모델 훈련
kn.fit(train_input, train_target)
##정확도0 = 샘플링 편향 ,치우쳐졋기때문에
##samplinng bias
kn.score(test_input, test_target)
#Own Dimension:벡터
#Two Dimension :matrix
import numpy as np

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

print(input_arr)
print(input_arr.shape)#샘플수,특성수 
#
np.random.seed(42)
index = np.arange(49)#0부터 48까지 1씩 증가하는 인덱스
np.random.shuffle(index)#  shuffle

print(index)

print(input_arr[[1,3]])

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

print(input_arr[13], train_input[0])

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

import matplotlib.pyplot as plt

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
#섞은거 확인
plt.show()
#훈련
kn.fit(train_input, train_target)
#테스트
kn.score(test_input, test_target)
#테스트 세트
kn.predict(test_input)

test_target

#지도학습은 입력과 타깃을 입력하여 새로운데이터를 예측하는데 활용
#비지도학습은 타깃데이터가 없으므로 특징을 찾는데 주로활용
#훈련세트 :모델을 훈련훈련세트가 클수록 좋음
#테스트세트:전체데이터에서 20%사용하는것이 좋음
#seed:넘파이에서 난수생성
#arange()일정한 간격의 정수 또는 실수배열