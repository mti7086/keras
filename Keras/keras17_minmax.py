from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 전처리 1
# minmaxscaler란 최소값(Min)과 최대값(Max)을 사용해서 '0~1' 사이의 범위(range)로 데이터를 표준화해주는 '0~1 변환'
# X_MinMax = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))


#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20000,30000,40000],[30000,40000,50000], [40000,50000,60000], 
            [100,200,300]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x) # evaluate predict 하는 과정

# scaler = StandardScaler()
# scaler.fit(x)
# x = scaler.transform(x) # evaluate predict 하는 과정

scaler = RobustScaler()
scaler.fit(x)
x = scaler.transform(x)

# train과 predict로 나눌 것
# train = 1번째부터 13번째까지 
# predict = 14번째

x_train = x[:13, :]
y_train = y[:13]
x_predict = x[13:14, :]

print(x_train.shape)
print(y.shape)
print(y_train.shape)
print(x_predict.shape)
print(x_predict)

# x = (4, 3) ↓
#      ↑x.shape[0]
# x = x.reshape((x.shape[0], x.shape[1], 1)) # 1개씩 자르는 걸 명시하기 위해 reshape를 함
# x = x.reshape((x.shape[0], x.shape[1], 1)) reshape 강제변환같이 생각해도 되겠네
# x =          (     13     ,     3    , 1  )

#2. 모델 구성
model = Sequential()
model.add(Dense(20, activation='relu', input_shape=(3, ))) # input_shape = (행, 3(열), 1->1개씩 자름) 
# activation 기본값은 activation='linear'
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))
model.summary()

#3. 실행
import numpy as np
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1, verbose=1) # verbose=0하면 결과만 나옴
# verbose=1 하면 훈련하는 과정을 보여줌
# verbose=2 하면 간략하게 훈련 과정을 보여줌

# x_input = array([25,35,45]) # (1, 3, ?????)
# x_input = np.transpose(x_input)
# # x_input = scaler.transform(x_input)

# print(x_input.shape)

# x_input = x_input.reshape((1,3))
# x_input = x_input.reshape((1,3,1)) # (1->행,3->열,1->1개씩 자름)

yhat = model.predict(x_predict, verbose=1)
print(yhat)
