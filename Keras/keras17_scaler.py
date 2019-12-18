from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 전처리 2
# minmaxscaler란 최소값(Min)과 최대값(Max)을 사용해서 '0~1' 사이의 범위(range)로 데이터를 표준화해주는 '0~1 변환'
# X_MinMax = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# StandardScaler란 평균이 0과 표준편차가 1이 되도록 변환.

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20000,30000,40000],[30000,40000,50000], [40000,50000,60000]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000])

from sklearn.preprocessing import MinMaxScaler, StandardScaler # 전처리과정
# ---------------MinMaxScaler()----------------------
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x) # evaluate predict 하는 과정
# print(x)

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x) # evaluate predict 하는 과정
print(x)


print("x.shape : ", x.shape) # (13, 3)
print("y.shape : ", y.shape) # (13,) 벡터가 13개

# x = (4, 3) ↓
#      ↑x.shape[0]
# x = x.reshape((x.shape[0], x.shape[1], 1)) # 1개씩 자르는 걸 명시하기 위해 reshape를 함
# x = x.reshape((x.shape[0], x.shape[1], 1)) reshape 강제변환같이 생각해도 되겠네
# x =          (     13     ,     3    , 1  )

print("x.shape : ", x.shape) # (13(행은무시), 3, 1->1개씩 자름) 

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(3, ))) # input_shape = (행, 3(열), 1->1개씩 자름) 
# activation 기본값은 activation='linear'
model.add(Dense(5))
model.add(Dense(1))
model.summary()

#3. 실행
import numpy as np
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1000, batch_size=1, verbose=0) # verbose=0하면 결과만 나옴
# verbose=1 하면 훈련하는 과정을 보여줌
# verbose=2 하면 간략하게 훈련 과정을 보여줌

x_input = array([25,35,45]) # (1, 3, ?????)
x_input = np.transpose(x_input)
# x_input = scaler.transform(x_input)

print(x_input.shape)

# x_input = x_input.reshape((1,3))
# x_input = x_input.reshape((1,3,1)) # (1->행,3->열,1->1개씩 자름)

# yhat = model.predict(x_input)
# print(yhat)