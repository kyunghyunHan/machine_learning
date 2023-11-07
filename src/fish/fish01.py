


# 훈련 셋, 테스트 셋
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


import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)
plt.xlabel('length')#x축은 길이
plt.ylabel('weight')#y축은 무게
plt.show()
## 빙어데이터 추가
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
##리스트합치기
length = bream_length+smelt_length
weight = bream_weight+smelt_weight

##2차원 리스트 만들기
fish_data = [[l, w] for l, w in zip(length, weight)]

print(fish_data)
#정답
##머신러닝에서 2개를 구분하는 경우 찾을려는 대상을 1 그외에는 0 도미는 1 빙어는 0
fish_target = [1]*35 + [0]*14
print(fish_target)

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
# 주어진 데이터로 훈련
kn.fit(fish_data, fish_target)
# 정확도
kn.score(fish_data, fish_target)

#k-최근접 이웃 알고리즘
#주위의 다른 데이터를 보고 다수를 차지하는 것을 정답으로 사용


plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
## 삼각형
## predict:새로운 데이터의 정답을 예측
kn.predict([[30, 600]])
## 도미는 1
print(kn._fit_X)# array([1])

print(kn._y)

##기본값은 5
kn49 = KNeighborsClassifier(n_neighbors=49)

kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)

print(35/49)

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
# 정확도= (정확히 맞힌 개수)/(전체 데이터 개수)
for n in range(5, 50):
    # 최근접 이웃 개수 설정
    kn.n_neighbors = n
    # 점수 계산
    score = kn.score(fish_data, fish_target)
    # 100% 정확도에 미치지 못하는 이웃 개수 출력
    if score < 1:
        print(n, score)
        break