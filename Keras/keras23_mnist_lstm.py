from keras.datasets import mnist
# mnist는 데이터 세트에 대한 간단한 컨 버넷을 훈련시킵니다.]
# 분류기반 시작
# 원-핫 인코딩(One-hot encoding)은 자연어 처리에서는 문자를 숫자(벡터)로 바꾸는 여러가지 기법 중 하나

# 원-핫 인코딩(One-hot encoding)은 단어 집합의 크기를 벡터의 차원으로 하고, 
# 표현하고 싶은 단어의 인덱스에 1의 값을 부여(쓰고 싶은 것만)하고, 
# 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식입니다.

(X_train, Y_train), (X_test, Y_test) = mnist.load_data() # 훈련용 데이터 다운

print(X_train[0])
print(Y_test[0])
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import tensorflow as tf

X_train = X_train.reshape(X_train.shape[0], 28*28, 1).astype('float32')/ 255
# (60000, 28, 28, 1)
# 255로 나눈 이유는 Max 값이 255였으니, MinMaxScaler로 전처리 한것이다.
X_test = X_test.reshape(X_test.shape[0], 28*28,1 ).astype('float32')/ 255
# (10000, 28, 28, 1)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_train[0])
print(Y_train.shape)
print(Y_test.shape)

# 컴볼루션 신경망의 설정
model = Sequential()

# 28 -3+1 = 26 
model.add(LSTM(32, input_shape=(28*28, 1), activation='relu'))
model.add(Dense(64, activation='relu')) # 26-3+1=24
# model.add(MaxPooling2D(pool_size=2)) # pool_size=2->(2,2)에서 가장 큰 것을 빼겠다.
# (None, 12,12,64)
# 전(24,24,64)에서 MaxPooling2D(pool_size=2)하면 24/2=12로 (12,12,64)가 된다.
model.add(Dropout(0.25))
# model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# y calum이 10이 되었다. 10개의 calum 중 1개를 고른다. softmax => 분류모델
model.summary()
 
# 분류 모델 categorical_crossentropy 다중 분류 손실함수
model.compile(loss='categorical_crossentropy'
                ,optimizer='adam'
                ,metrics=['accuracy'])

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=2, batch_size=200, verbose=1, # epochs=30
                    callbacks=[early_stopping_callback]) #,checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test,Y_test)[1]))
# # model.compile(loss='categorical_crossentropy' [0]           ↑
#                 ,optimizer='adam'                             ↑
#                 ,metrics=['accuracy']) [1]--------------------↑