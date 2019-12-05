# import keras # keras 가져오겠다.
# import tensorflow

#1. 데이터
import numpy as np # numpy를 가져와 np로 지정해주겠다.

x = np.array(range(1, 101))
y = np.array(range(1, 101))
print(x)

x_train = x[:60] # 1부터 60까지 자름
x_val = x[60:80] # 61부터 80까지 자름 / 20개를 validation함(검증)
x_test = x[80:] # 81부터 100까지 자름

y_train = y[:60] # 1부터 60까지 자름
y_val = y[60:80] # 61부터 80까지 자름 / 20개를 validation함(검증)
y_test = y[80:] # 81부터 100까지 자름
# 데이터 정제하고 있다. 현재 데이터를 자른 상태

#2. 모델구성
from keras.models import Sequential # keras의 models폴더안에 Sequential만 가져오겠다.
from keras.layers import Dense
model = Sequential() # 순차적으로~
# dimension: 차원, input_dim=1 1차원?

# model.add(Dense(40, input_dim=1, activation='relu')) # input_dim=1 1개를 받아서 Dense(5,) 5개를 출력
model.add(Dense(5, input_shape=(1, ), activation='relu')) # input_shape=1 1개를 받아서 Dense(5,) 5개를 출력
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

# model.summary() # 요약

#3. 훈련
model.compile(loss='mse', optimizer='adam', # metrics=['accuracy']) # loss=손실함수 optimizer=최적화
                metrics=['mse']) # loss: 6.8923e-13, metrics->mse: 6.8923e-13
# / 이 모듈에서 최소한으로 손실보겠다. accuracy 정확성
# model.fit(x_train, y_train, epochs=500, batch_size=1) # fit == 트레이닝 100번 훈련 1,2,3,4,5를 한개씩(1) 잘라서 훈련
model.fit(x_train, y_train, epochs=100, batch_size=1,
        validation_data=(x_val, y_val)) # validation_data=머신한테 니가 검증해가면서 학습하라는 뜻
# batch_size란 sample데이터 중 한번에 네트워크에 넘겨주는 데이터의 수를 말한다.
# 1:1 과외가 공부 잘되는 것 처럼 1억개 데이터이면 100만개씩 모아서 공부시키듯이

#4. 평가 예측
# mse = 평균 제곱 에러 오답에 가까울수록 큰 값이 나온다. 반대로 정답에 가까울수록 작은 값이 나온다.
loss, acc = model.evaluate(x_test, y_test, batch_size=3) # evaluate함수에 x_test, y_test값을 넣으면 loss a[0], acc a[1] 값이 나온다.
print("acc : ", acc)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기 RMSE = MSE에 루트를 씌워준 것
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) # mse = mean squared error/sqrt=루트값
    # y_predict와 주어진 값 y_test와 비교
print("RMSE: ", RMSE(y_test, y_predict))

# R2 구하기 R2는 1이 나오면 잘한거 0이 나오면 최악이라는 지표일 뿐 0.9999 잘 됬다는 것은 아니다.
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

# R2 0.9999999999988425 RMSE 3.0902579776801397e-06
# R2의 값을 늘렸더니 R2 0.9999999999992074 늘고 , RMSE의 값은 낮아졌다. RMSE 3.846355047163435e-06