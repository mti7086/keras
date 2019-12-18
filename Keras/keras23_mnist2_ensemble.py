# x_train (60000, 28, 28) -> x1 , x2 각 3만
# y도 통일

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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import tensorflow as tf

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')/ 255
# (60000, 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')/ 255
# (10000, 28, 28, 1)

# X1_train = X_train[:30000, :, :]
# X2_train = X_train[30000:, :, :]
# print(X1_train)
# print(X1_train.shape)
# print(X2_train.shape)
# print(Y_train.shape)

# Y1_train = Y_train[:30000]
# Y2_train = Y_train[30000:]

from sklearn.model_selection import train_test_split
X1_train, X2_train, Y1_train, Y2_train = train_test_split(
    X_train, Y_train, random_state=50, train_size=0.5, shuffle=False
) # x 값을 x_train, x_test로 분류 y 값을 y_train, y_test로 분류

Y1_train = np_utils.to_categorical(Y1_train)
Y2_train = np_utils.to_categorical(Y2_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_train[0])
print(Y_train.shape)
print(Y1_train.shape)
print(Y2_train.shape)
print(Y_test.shape)

# 
# 컴볼루션 신경망의 설정
# model = Sequential()

# 28 -3+1 = 26 
# model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
# model.add(Conv2D(64, (3,3), activation='relu')) # 26-3+1=24
# model.add(MaxPooling2D(pool_size=2)) # pool_size=2->(2,2)에서 가장 큰 것을 빼겠다.
# # (None, 12,12,64)
# # 전(24,24,64)에서 MaxPooling2D(pool_size=2)하면 24/2=12로 (12,12,64)가 된다.
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))
# # y calum이 10이 되었다. 10개의 calum 중 1개를 고른다. softmax => 분류모델
# model.summary()

input1 = Input(shape=(28,28,1))
xx = Conv2D(64, (3,3), activation='relu')(input1)
xx = Flatten()(xx)
xx = Dense(30)(xx)
middle1 = Dense(28)(xx)

input2 = Input(shape=(28,28,1))
xx = Conv2D(64, (3,3), activation='relu')(input2)
xx = Flatten()(xx)
xx = Dense(30)(xx)
middle2 = Dense(28)(xx)

from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])

output1 = Dense(30)(merge1)
output1 = Dense(13)(output1)
output1 = Dense(10, activation='softmax')(output1) # output shape(1,) 에러뜨면 여길 고치자

output2 = Dense(15)(merge1)
output2 = Dense(32)(output2)
output2 = Dense(10, activation='softmax')(output2)

model = Model(inputs = [input1,input2], outputs = [output1,output2]) # 어디서부터 어디까지 model이라는 걸 선언
model.summary() # 요약


# 분류 모델 categorical_crossentropy 다중 분류 손실함수
model.compile(loss='categorical_crossentropy'
                ,optimizer='adam'
                ,metrics=['accuracy'])

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit([X1_train,X2_train], [Y1_train,Y2_train], validation_data=(X_test, Y_test),
                    epochs=2, batch_size=200, verbose=1, # epochs=30
                    callbacks=[early_stopping_callback]) #,checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test,Y_test)[1]))
# # model.compile(loss='categorical_crossentropy' [0]           ↑
#                 ,optimizer='adam'                             ↑
#                 ,metrics=['accuracy']) [1]--------------------↑
