from keras.models import Sequential # keras의 models폴더안에 Sequential만 가져오겠다.
from keras.layers import Dense

# import keras # keras 가져오겠다.
# import tensorflow

import numpy as np # numpy를 가져와 np로 지정해주겠다.
x = np.array([1,2,3,4,5]) # 정제된 데이터
y = np.array([1,2,3,3.5,5]) # 정제된 데이터

model = Sequential() # 순차적으로~
model.add(Dense(5, input_dim=1, activation='relu')) # input_dim=1 1개를 받아서 Dense(5,) 5개를 출력
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam') # loss=손실함수 optimizer=최적화/ 이 모듈에서 최소한으로 손실보겠다.
model.fit(x, y, epochs=100, batch_size=1) # fit == 트레이닝 100번 훈련/ 1,2,3,4,5를 한개씩(1) 잘라서 훈련
# batch_size란 sample데이터 중 한번에 네트워크에 넘겨주는 데이터의 수를 말한다.

# mse = 평균 제곱 에러 오답에 가까울수록 큰 값이 나온다. 반대로 정답에 가까울수록 작은 값이 나온다.
mse = model.evaluate(x, y, batch_size=1) # 평가
print("mse : ", mse)

