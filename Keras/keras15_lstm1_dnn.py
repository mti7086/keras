import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11)) # 정제되지 않은 데이터
size =5
def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):# (6,5) 행
        subset = seq[i:(i+size)] # (6,5) 열
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("==================")
print(dataset)

x_train = dataset[:, 0:-1] # [:, 0:4] == (행,열)
y_train = dataset[:, -1] # [:, 0:4] == (행, 열)

print(x_train.shape) # (6, 4)
print(x_train)
print(y_train.shape) # (6, )
print(y_train)

model = Sequential()
model.add(Dense(20, input_shape=(4, ), activation='relu'))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(1)) # calum수 1개

from keras.callbacks import EarlyStopping
model.compile(loss='mse', optimizer='adam', # metrics=['accuracy']) # loss=손실함수 optimizer=최적화
                metrics=['mse'])
early_stopping = EarlyStopping(monitor='mse', patience=100, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=1, callbacks=[early_stopping])

x2 = np.array([7,8,9,10]) # (4, ) -> (1, 4)
x2 = x2.reshape((1,4))

y_pred = model.predict(x2)
print(y_pred)
