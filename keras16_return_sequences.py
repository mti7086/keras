from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40],[30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("x.shape : ", x.shape) # (13, 3)
print("y.shape : ", y.shape) # (13,) 벡터가 13개

# x = (4, 3) ↓
#      ↑x.shape[0]
x = x.reshape((x.shape[0], x.shape[1], 1)) # 1개씩 자르는 걸 명시하기 위해 reshape를 함
# x = x.reshape((x.shape[0], x.shape[1], 1)) reshape 강제변환같이 생각해도 되겠네
# x =          (     13     ,     3    , 1  )

print("x.shape : ", x.shape) # (13(행은무시), 3, 1->1개씩 자름) 

#2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1), 
               return_sequences=True))
model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(LSTM(10, activation='relu', return_sequences=True)) 
model.add(LSTM(10, activation='relu', return_sequences=True)) 
model.add(LSTM(10, activation='relu', return_sequences=True))  
model.add(LSTM(10, activation='relu', return_sequences=True)) 
model.add(LSTM(10, activation='relu', return_sequences=True)) 
model.add(LSTM(10, activation='relu', return_sequences=True)) 
model.add(LSTM(10, activation='relu', return_sequences=True)) 
model.add(LSTM(10, activation='relu', return_sequences=True)) 
# ↑ input_shape = (행, 3(열), 1->1개씩 자름) 
# 에러 이유 = input: (None, 3, 1) ndim=3 -> output(None, 10) ndim=2
model.add(LSTM(3)) # -> input: (None, 10) ndim=2
model.add(Dense(5))
model.add(Dense(1))
model.summary()


#3. 실행
from keras.callbacks import EarlyStopping

model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
# monitor='loss'값을 모니터 해서 원하는 loss값(최저값)이 나오는데 patience=100-> 100번 나오면 끝냄
# mode='auto' 자동으로 하겠다 
model.fit(x, y, epochs=1000, callbacks=[early_stopping]) # verbose=0하면 결과만 나옴

# model.fit(x, y, epochs=1000, batch_size=1, verbose=0)
# verbose=1 하면 훈련하는 과정을 보여줌
# verbose=2 하면 간략하게 훈련 과정을 보여줌

x_input = array([25,35,45]) # (1, 3, ?????)
x_input = x_input.reshape((1,3,1)) # (1->행,3->열,1->1개씩 자름)

yhat = model.predict(x_input)
print(yhat)
