from keras.models import Sequential # keras의 models폴더안에 Sequential만 가져오겠다.
from keras.layers import Dense

# import keras # keras 가져오겠다.
# import tensorflow


import numpy as np # numpy를 가져와 np로 지정해주겠다.
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 정제된 데이터/훈련 데이터
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 정제된 데이터/훈련 데이터
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
x_predict = np.array([21,22,23,24,25])

model = Sequential() # 순차적으로~
# dimension: 차원, input_dim=1 1차원?
# model.add(Dense(40, input_dim=1, activation='relu')) # input_dim=1 1개를 받아서 Dense(5,) 5개를 출력
model.add(Dense(40, input_shape=(1, ), activation='relu')) # input_shape=1 1개를 받아서 Dense(5,) 5개를 출력
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.summary() # 요약

model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # loss=손실함수 optimizer=최적화
# / 이 모듈에서 최소한으로 손실보겠다. accuracy 정확성
model.fit(x_train, y_train, epochs=1000, batch_size=5) # fit == 트레이닝 100번 훈련 1,2,3,4,5를 한개씩(1) 잘라서 훈련
# batch_size란 sample데이터 중 한번에 네트워크에 넘겨주는 데이터의 수를 말한다.
# 1:1 과외가 공부 잘되는 것 처럼 1억개 데이터이면 100만개씩 모아서 공부시키듯이

# mse = 평균 제곱 에러 오답에 가까울수록 큰 값이 나온다. 반대로 정답에 가까울수록 작은 값이 나온다.
loss, acc = model.evaluate(x_test, y_test, batch_size=5) # 평가
print("acc : ", acc)
print("loss : ", loss)

y_predict = model.predict(x_predict)
print(y_predict)