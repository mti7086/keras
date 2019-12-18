# earlyStopping 적용하기 실습,
# loss, acc, val_loss, val_acc
# keras05.py를 카피해서 사용
# split사용 하시라고 언뜻 애기하셨음
# keras15_lstm2.py


from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

# import keras # keras 가져오겠다.
# import tensorflow


import numpy as np # numpy를 가져와 np로 지정해주겠다.
x_train = np.array([[1,2,3,4,5,6,7,8,9,10]]) # 정제된 데이터/훈련 데이터
y_train = np.array([[1,2,3,4,5,6,7,8,9,10]]) # 정제된 데이터/훈련 데이터
x_test = np.array([[11,12,13,14,15,16,17,18,19,20]])
y_test = np.array([[11,12,13,14,15,16,17,18,19,20]])
x_predict = np.array([21,22,23,24,25])

from sklearn.model_selection import train_test_split
x1_train, x2_train, y1_train, y2_train = train_test_split(
    x_train, y_train, random_state=50, x1_train_size=0.5, shuffle=False
) # x 값을 x_train, x_test로 분류 y 값을 y_train, y_test로 분류
# x_val, x_test, y_val, y_test = train_test_split(
#     x_test, y_test, random_state=50, test_size=0.4, shuffle=False # val 20% test 20% train 60%
# ) # x_test 값을 x_val, x_test로 분류 y_test 값을 y_val, y_test로 분류
# # test를 40% 주겠다. train을 60%주겠다

# x1_train, x2_train = x_train[:5, :5], x_train[:9, 5:]
# y1_train, y2_train = y_train[:5, :5], y_train[:9, 5:]
# x1_test, x2_test = x_test[:5, :5], x_test[:9, 5:]
# y1_test, y2_test = y_test[:5, :5], y_test[:9, 5:]

# print(x1_train, x2_train)
# print(y1_train, y2_train)
# print(x1_test, x2_test)
# print(y1_test, y2_test)

x1_train = x1_train.reshape((1, 5, 1))
x2_train = x2_train.reshape((1, 5, 1))
x1_test = x1_test.reshape((1,5,1))
x2_test = x2_test.reshape((1,5,1))
y1_test = y1_test.reshape((1,5))
y2_test = y2_test.reshape((1,5))
y1_train = y1_train.reshape((1,5))
y2_train = y2_train.reshape((1,5))

print(x1_train.shape) 
print(x2_train.shape) 
print(y1_train.shape)
print(y2_train.shape)
print(x1_test.shape)
print(x2_test.shape)
print(y1_test.shape)
print(y2_test.shape)

"""
model = Sequential() # 순차적으로~
# dimension: 차원, input_dim=1 1차원?
# model.add(Dense(40, input_dim=1, activation='relu')) # input_dim=1 1개를 받아서 Dense(5,) 5개를 출력
model.add(LSTM(10, activation='relu', input_shape=(5,1)))# input_shape=1 1개를 받아서 Dense(5,) 5개를 출력
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(2))

model.summary() # 요약

model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')# loss=손실함수 optimizer=최적화
# / 이 모듈에서 최소한으로 손실보겠다. accuracy 정확성
model.fit([x1_train,x2_train],[y1_train,y2_train], epochs=1, callbacks=[early_stopping]) # fit == 트레이닝 100번 훈련 1,2,3,4,5를 한개씩(1) 잘라서 훈련
# batch_size란 sample데이터 중 한번에 네트워크에 넘겨주는 데이터의 수를 말한다.
# 1:1 과외가 공부 잘되는 것 처럼 1억개 데이터이면 100만개씩 모아서 공부시키듯이

# mse = 평균 제곱 에러 오답에 가까울수록 큰 값이 나온다. 반대로 정답에 가까울수록 작은 값이 나온다.
loss = model.evaluate([x1_test,x2_test], [y1_test,y2_test], batch_size=1) # 평가
print("loss : ", loss)

x_predict = x_predict.reshape((1,5, 1))
y_predict = model.predict(x_predict)
print(y_predict)
"""