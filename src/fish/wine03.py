#정형데이터
#비정형데이터:텍스트,카메라사진,핸드폰      
#정형데이터를 다루는데 가장 뛰어난 성과를 내는 알고리즘이 앙상블학습
#비정형알고리즘->신경망알고리즘

#랜덤포레스트:결정트리를 랜덤하게 만들어 결정트리의 숲을 만듭니다
#각 결정 트리의 예측을 사용해 최족예측을 만듭니다

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data= wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()

train_input,test_input,train_target,test_target= train_test_split(
    data,target,test_size=0.2 ,random_state= 42
)

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1,random_state=42)

scores= cross_validate(rf,train_input,train_target,return_train_score=True,n_jobs=-1)

print(np.mean(scores['train_score']),np.mean(scores['test_score']))


rf.fit(train_input,train_target)
print(rf.feature_importances_)

rf= RandomForestClassifier(oob_score= True,n_jobs=-1,random_state=42)
rf.fit(train_input,train_target)
print(rf.oob_score_)

#gradient boosting

from sklearn.ensemble import GradientBoostingClassifier
gb= GradientBoostingClassifier(n_estimators=500,learning_rate=0.2,random_state=42)

scores=cross_validate(gb,train_input,train_target,return_train_score=True,n_jobs=-1)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))


gb.fit(train_input,train_target)
print(gb.feature_importances_)

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb= HistGradientBoostingClassifier(random_state=42)
scores= cross_validate(hgb,train_input,train_target,return_train_score=True)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))

from sklearn.inspection import permutation_importance

hgb.fit(train_input,train_target)

result = permutation_importance(hgb,train_input,train_target,n_repeats=10,random_state=42,n_jobs=-1)
print(result.importances_mean)

result= permutation_importance(hgb,test_input,test_target,n_repeats=10,random_state=42,n_jobs=-1)

print(result.importances_mean)

hgb.score(test_input,test_target)
