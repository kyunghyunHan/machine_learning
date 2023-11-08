import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
print(pd.unique(fish['Species']))

fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
print(fish_input[:5])

fish_target = fish['Species'].to_numpy()


from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target= train_test_split(
    fish_input,fish_target, random_state= 42
)

#전처리
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled=ss.transform(train_input)
test_scaled=ss.transform(test_input)

# 최근접 이웃 분류기의 확률 예측
from sklearn.neighbors import KNeighborsClassifier

kn= KNeighborsClassifier()
kn.fit(train_scaled,train_target)
print(kn.score(train_scaled,train_target))
print(kn.score(test_scaled,test_target))

print(kn.classes_)
print(kn.predict(test_scaled[:5]))

import numpy as np
proba= kn.predict_proba(test_scaled[:5])
print(np.round(proba,decimals=4))

distances,indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])
# 로지스틱 회귀
import numpy as np
import matplotlib.pyplot as plt
z= np.arange(-5,5,0.1)
phi= 1/(1+np.exp(-z))
plt.plot(z,phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()
#로지스틱 회귀로 이진분류 수행
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

print(lr.predict(train_bream_smelt[:5]))

print(lr.predict_proba(train_bream_smelt[:5]))

print(lr.classes_)

print(lr.coef_, lr.intercept_)

decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

from scipy.special import expit

print(expit(decisions))
# 로지스틱 회귀로 다중 분류 수행
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.predict(test_scaled[:5]))

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

print(lr.classes_)

print(lr.coef_.shape, lr.intercept_.shape)
#z1~z7까징의 값을 구한다음
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

from scipy.special import softmax
#softmax의 axis매개변수는 소프트맥스를 계산할 축을 지정 1로 지정하여 각 행,각 샘플에 대해 소프트맥스 계산 axis를 지정하지 않음녀 배열전체에대해 계산
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
# k-최근접 이웃모델이 확률을 출력할수 있지만 이웃한 샘플의 클래스비율이므로 항상 정해진 확률만 출력

# 로지스틱 회귀(분류모델)는 이진 분류에서는 하나의 선형방정식을 훈련 
# 출력값을 시그모이드 함수에 통과시켜 0에서 1사이의 값을 생성,이 값이 양성 클래스에 대한 확률,음성 클래스의 확률은 1에서 양성클래스의 확률 뺴기
# 다중 분류일 경우 클래스 개수만큼 방정식 훈련
# 그다음 각 방정식의 출력값을 소프트맥스 함수를 통과시켜 전체 클래스의 합이 항상 1이되도록
