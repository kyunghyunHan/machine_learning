#와인분류
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')

wine.head()
# 데이터 프레임의 각 열의 데이터 타입과 누락된 데이터가 있는지 확인하는데 유용
wine.info()

# 열에 대한 간략한 통계출력
# 평균:mean 표준편차:std min,max
wine.describe()

data= wine[['alcohol','sugar','pH']].to_numpy()
target= wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target= train_test_split(
    data,target,test_size=0.2 ,random_state= 42
)

print(train_input.shape,test_input.shape)

#표준화 전처리
from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
ss.fit(train_input)
train_scaled= ss.transform(train_input)
test_scaled= ss.transform(test_input)



from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled,train_target)
print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))
#과소적합

#결정트리

#과대적합
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(train_scaled,train_target)
print(dt.score(train_scaled,train_target))
print(dt.score(test_scaled,test_target))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()

plt.figure(figsize=(10,7))
plot_tree(dt,max_depth=1,filled=True,feature_names=['alcohol','sugar','pH'])
plt.show()


dt= DecisionTreeClassifier(max_depth=3,random_state=42)
dt.fit(train_scaled,train_target)
print(dt.score(train_scaled,train_target))
print(dt.score(test_scaled,test_target))


plt.figure(figsize=(20,15))
plot_tree(dt,filled=True,feature_names=['alcohol','sugar','pH'])
plt.show()

print(dt.feature_importances_)



