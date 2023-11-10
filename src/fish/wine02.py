#검증세트
#테스트세트를 하지않으면 모델이 과대적합인지 과소적합인지 판단하기 어려움
#테스트세트를 사용하지 않고 측정하는 방법은 훈련세트를 나누면댐 이데이터를 검증세트

import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')

data= wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target= train_test_split(
    data,target,test_size=0.2 ,random_state= 42
)

sub_input,val_input,sub_target,val_target= train_test_split(
    train_input,train_target,test_size=0.2 ,random_state= 42
)

print(sub_input.shape,val_input.shape)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(sub_input,sub_target)
print(dt.score(sub_input,sub_target))
print(dt.score(val_input,val_target))

#교차 검증을 이용하면 안정적인 검증 점수를 얻고 훈련에 더 많은 데이터 사용가능

from sklearn.model_selection import cross_validate
scores= cross_validate(dt,train_input,train_target)
print(scores)

#처음 2개는 각각 모델을 훈련하는 시간과 검증하는 시간을 의미
#교차 검증의 최종 점수는 test_score킹 담긴 5개의 점수를 평균해서 얻을수 있음
import numpy as np
print(np.mean(scores['test_score']))

from sklearn.model_selection import StratifiedKFold
scores= cross_validate(dt,train_input,train_target,cv= StratifiedKFold())
print(np.mean(scores['test_score']))


#훈련세트를 섞은 후 10-폴드 교차 검증을 수해하려면 
splitter= StratifiedKFold(n_splits= 10,shuffle=True,random_state=42)
scores= cross_validate(dt,train_input,train_target,cv= splitter)
print(np.mean(scores['test_score']))

#하이퍼 파라미터 튜닝
#머신러닝 모델이 학습하는 파라미터를 모델파라미터
#모델이 학습할수 없어서 사용자가 지정해야만 하는 파라미터를 하이퍼파라미터

#그리드서치:하이퍼 파라미터 탐색과 교차 검증을 한번에 수행

from sklearn.model_selection import GridSearchCV
params={'min_impurity_decrease':[0.0001,0.0002,0.0003,0.0004,0.0005]}

gs=GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1)
gs.fit(train_input,train_target)
dt= gs.best_estimator_
print(dt.score(train_input,train_target))

print(gs.best_params_)

print(gs.cv_results_['mean_test_score'])


best_index= np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])

#탐색할 매개변수를 지정
#훈련세트에서 그리드 서치를 수행하여 최상의 평균검증점수가 나오는 매개변수 조합을 찾는다, 이조합은ㅁ 그리드 서치 객체에 저장
#그리드 서치는 최상의 매개변수에서 (교차검증에 사용한 훈련세트가 아니라)전체 훈련새트를 사용해 최종 모델을 훈련,이모델도 그리드 서치 객체에 저장

params={'min_impurity_decrease':np.arange(0.0001,0.001,0.0001),
        'max_depth':range(5,20,1),
        'min_samples_split':range(2,100,10)}
gs=GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=1)
gs.fit(train_input,train_target)
print(gs.best_params_)

print(np.max(gs.cv_results_['mean_test_score']))


from scipy.stats import uniform, randint

rgen=  randint(0,10)
rgen.rvs(10)

np.unique(rgen.rvs(1000),return_counts= True)

ugen= uniform(0,1)
ugen.rvs(10)


params={'min_impurity_decrease':uniform(0.0001,0.001),
        'max_depth':randint(20,50),
        'min_samples_split':randint(2,25),
         'min_samples_leaf':randint(1,25),
        }

from sklearn.model_selection import RandomizedSearchCV
gs= RandomizedSearchCV(DecisionTreeClassifier(random_state=42),params,
n_iter=100,n_jobs=-1,random_state=42)
gs.fit(train_input,train_target)

print(gs.best_params_)

print(np.max(gs.cv_results_['mean_test_score']))

dt= gs.best_estimator_
print(dt.score(test_input,test_target))