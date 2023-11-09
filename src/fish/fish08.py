#훈련한 모델을 버리지 않고 새로운 데이터에 대해서만 조금씩 훈련하는 방식을 점진적학습
#대표적인 확률적 경사 하강법
#확률적 경사 하강법에서 확률적이란 무작위하게또는 랜덤하게라는 기술적 표현
#훈련세트를 랜덤하게 하나의 샘플을 고르는 것
#확률적 경사 하강법에서 훈련세트를 한번 모두 사용하는 것을 에포크 라고함
#여러개의 샘프을 사용해서 경사하강법을 수행하는 방식을 미니배치 경사 하강법이라합니다.
#확률적 경사 하강법을 꼭 사용하는 알고리즘은 신경망 알고리즘 

#손실함수
#손실함수는 어떤 문제에서 머신러닝 알고리즘이 얼마나 엉터리인지 측정하는 기준

#분류에서 손실은 정답을 못맞히는것

#로지스틱 손실함수또는 이진크로스엔트로피 손실함수 :사용하여 로지스틱 회귀모델을 만듬

#크로스렌트로피 손실함수:다중분류에서 사용하는 손실함수

#회귀에는 평균제곱오차 사용

import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
print(pd.unique(fish['Species']))


fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
print(fish_input[:5])
# 타겟
fish_target = fish['Species'].to_numpy()

#훈련세트와 테스트 세트로 나누기
from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target= train_test_split(
    fish_input,fish_target, random_state= 42
)

#표준화 전처리
from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
ss.fit(train_input)
train_scaled= ss.transform(train_input)
test_scaled= ss.transform(test_input)

from sklearn.linear_model import SGDClassifier

sc= SGDClassifier(loss= 'log', max_iter=10,random_state=42)
sc.fit(train_scaled,train_target)
print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,train_target))

#sc_partial_fit() :호출마다 1에포크씩 이어서 훈련가능

sc.partial_fit(train_scaled,train_target)
print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))

#에포크와 과대/과소적합

import numpy as np
sc= SGDClassifier(loss='log',random_state=42)
train_score=[]
test_score=[]

classes= np.unique(train_target)

for _in range(0,300):
    sc.partial_fit(train_scaled,train_target,classes=classes)
    train_score.append(sc.score(train_scaled,train_target))
    train_score.append(sc.score(test_scaled,test_target))

import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


sc= SGDClassifier(loss='log',max_iter=100,tol=None,random_state=42)
sc.fit(train_scaled,train_target)

print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))

sc= SGDClassifier(loss='log',max_iter=100,tol=None,random_state=42)
sc.fit(train_scaled,train_target)

print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))