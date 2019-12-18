# keras13_lstm4를 카피해서
# x와 y 데이터를 각각 2개로 분리
# 2개의 인풋, 2개의 아웃풋 모델인 ensemble모델을 구현하시오.


from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40],[30,40,50], [40,50,60]])           
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("x.shape : ", x.shape) # (13, 3)
print("y.shape : ", y.shape) # (13,) 벡터가 13개

x1 = x[:10, :]
x2 = x[10:, :]
y1 = y[:10]
y2 = y[10:]

print(x1)
# x = (4, 3) ↓
#      ↑x.shape[0]
x1 = x1.reshape((x1.shape[0], x1.shape[1], 1))
x2 = x2.reshape((x2.shape[0], x2.shape[1], 1))
# 1개씩 자르는 걸 명시하기 위해 reshape를 함
# x = x.reshape((x.shape[0], x.shape[1], 1)) reshape 강제변환같이 생각해도 되겠네
# x =          (     13     ,     3    , 1  )

print("x1.shape : ", x1.shape) # (13(행은무시), 3, 1->1개씩 자름) 
print("y1.shape : ", y1.shape)
print("x2.shape : ", x2.shape)
print("y2.shape : ", y2.shape)


#2. 모델 구성
# model = Sequential()

input1 = Input(shape=(3,1))
xx = LSTM(40, activation='relu')(input1) # dense1 layer/ input: 1, ouput:5
xx = Dense(3)(xx) # dense2 layer/ input: 5, ouput:3
xx = Dense(4)(xx) # dense3 layer/ input: 3, ouput:4
xx = Dense(6)(xx)
xx = Dense(3)(xx)
xx = Dense(7)(xx)
xx = Dense(2)(xx)
xx = Dense(5)(xx)
middle1 = Dense(3)(xx)

# model.add(LSTM(40, activation='relu', input_shape=(10,1))) # input_shape=1 1개를 받아서 Dense(5,) 5개를 출력
# model.add(Dense(30))
# model.add(Dense(25))
# model.add(Dense(20))
# model.add(Dense(15))
# model.add(Dense(10))
# model.add(Dense(5))
# model.add(Dense(1))

input2 = Input(shape=(3,1))
xx = LSTM(40, activation='relu')(input2) # dense1 layer/ input: 1, ouput:5
xx = Dense(3)(xx) # dense2 layer/ input: 5, ouput:3
xx = Dense(4)(xx) # dense3 layer/ input: 3, ouput:4
xx = Dense(6)(xx)
xx = Dense(3)(xx)
xx = Dense(7)(xx)
xx = Dense(2)(xx)
xx = Dense(5)(xx)
middle2 = Dense(3)(xx)

from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])

output1 = Dense(30)(merge1)
output1 = Dense(13)(output1)
output1 = Dense(1)(output1) # output shape(1,) 에러뜨면 여길 고치자

output2 = Dense(15)(merge1)
output2 = Dense(32)(output2)
output2 = Dense(1)(output2)

model = Model(inputs = [input1,input2], outputs = [output1,output2]) # 어디서부터 어디까지 model이라는 걸 선언
model.summary() # 요약


#3. 실행
from keras.callbacks import EarlyStopping

model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
# monitor='loss'값을 모니터 해서 원하는 loss값(최저값)이 나오는데 patience=100-> 100번 나오면 끝냄
# mode='auto' 자동으로 하겠다 
# model.fit([x1,x2],[y1,y2], epochs=1000, callbacks=[early_stopping]) # verbose=0하면 결과만 나옴
# ValueError: All input arrays (x) should have the same number of samples. 
# Got array shapes: [(10, 3, 1), (3, 3, 1)] concatenate할 때 입력 data, 출력 data가 같아야함
model.fit([x1,x1],[y1,y1], epochs=1000, callbacks=[early_stopping])


# model.fit(x, y, epochs=1000, batch_size=1, verbose=0)
# verbose=1 하면 훈련하는 과정을 보여줌
# verbose=2 하면 간략하게 훈련 과정을 보여줌

# x1_input = array([25,35,45]) # (1, 3, ?????)
# x2_input = array([25,35,45])
# x1_input = x1_input.reshape((1,3,1)) # (1->행,3->열,1->1개씩 자름)
# x2_input = x2_input.reshape((1,3,1))

# y1hat, y2hat = model.predict([x1_input, x2_input])
# print(y1hat,y2hat)
