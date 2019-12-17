from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7])
print(x)

print("x.shape : ", x.shape) # (4, 3)
print("y.shape : ", y.shape) # (4,) 벡터가 4개


#  x            y
# [[[1],[2],[3]]4
# [[2],[3],[4]] 5
# [[3],[4],[5]] 6
# [[4],[5],[6]]]7

# x = (4, 3) ↓
#      ↑x.shape[0]
x = x.reshape((x.shape[0], x.shape[1], 1)) # 1개씩 자르는 걸 명시하기 위해 reshape를 함
# x = x.reshape((x.shape[0], x.shape[1], 1)) reshape 강제변환같이 생각해도 되겠네
# x =          (     4     ,     3    , 1  )

print("x.shape : ", x.shape) # (4(행은무시), 3, 1->1개씩 자름) 
print(x)

#2. 모델 구성
model = Sequential()
model.add(LSTM(15, activation='relu', input_shape=(3,1))) # input_shape = (행, 3(열), 1->1개씩 자름) 
model.add(Dense(14))
model.add(Dense(13))
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))
# model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1)

x_input = array([6,7,8]) # (1, 3, ?????)
x_input = x_input.reshape((1,3,1)) # (1->행,3->열,1->1개씩 자름)

yhat = model.predict(x_input)
print(yhat)