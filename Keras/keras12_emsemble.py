# import keras # keras 가져오겠다.
# import tensorflow

#1. 데이터
import numpy as np # numpy를 가져와 np로 지정해주겠다.

x1 = np.array([range(100), range(311,411), range(100)])
y1 = np.array([range(501, 601), range(711,811), range(100)])

x2 = np.array([range(101, 201), range(311,411), range(100, 200)])
y2 = np.array([range(501, 601), range(711,811), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)

print(x1.shape) # (100,3)
print(x2.shape) # (100,3)
print(y1.shape) # (100,3)
print(y2.shape) # (100,3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=33, test_size=0.4, shuffle=False
) # x 값을 x_train, x_test로 분류 y 값을 y_train, y_test로 분류
x1_val, x1_test, y1_val, y1_test = train_test_split(
    x1_test, y1_test, random_state=33, test_size=0.5, shuffle=False # val 20% test 20% train 60%
) # x_test 값을 x_val, x_test로 분류 y_test 값을 y_val, y_test로 분류
# test를 40% 주겠다. train을 60%주겠다
x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state=33, test_size=0.4, shuffle=False
)
x2_val, x2_test, y2_val, y2_test = train_test_split(
    x2_test, y2_test, random_state=33, test_size=0.5, shuffle=False # val 20% test 20% train 60%
)

print(x2_test.shape) #(20,3)
# random_state란
"""
random_state=50 순서는 50개 골라서 바뀌지만 데이터 분석하는데 문제 없다.
고정 된 random_state 일 때, 프로그램 실행마다 똑같은 결과를 산출합니다
random_state을 설정하지 않으면 알고리즘을 실행할 때마다 다른 시드가 사용되며 다른 결과가 나옵니다. 
x = (1, 2, 3, 4, 5) random_state하면 -> x = (1, 3, 4, 2, 5) 즉 순서는 바껴도 1이 1나오는 결과는 똑같다.
y = (1, 2, 3, 4, 5) random_state하면 -> x = (1, 3, 4, 2, 5) 즉 순서는 바껴도 1이 1나오는 결과는 똑같다.
"""


#2. 모델구성(함수형 모델)
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential() # 순차적모델

input1 = Input(shape=(3, )) # 최초의 input/ 컬럼 = 열 = 1개
dense1 = Dense(5, activation='relu')(input1) # dense1 layer/ input: 1, ouput:5
dense2 = Dense(3)(dense1) # dense2 layer/ input: 5, ouput:3
dense3 = Dense(4)(dense2) # dense3 layer/ input: 3, ouput:4
dense4 = Dense(6)(dense3)
dense5 = Dense(3)(dense4)
dense6 = Dense(7)(dense5)
dense7 = Dense(2)(dense6)
dense8 = Dense(3)(dense7)
dense9 = Dense(4)(dense8)
dense10 = Dense(10)(dense9)
dense11 = Dense(2)(dense10)
dense12 = Dense(7)(dense11)
dense13 = Dense(3)(dense12)
dense14 = Dense(12)(dense13)
dense15 = Dense(3)(dense14)
dense16 = Dense(7)(dense15)

middle1 = Dense(3)(dense16)

input2 = Input(shape=(3, )) # 최초의 input/ 컬럼 = 열 = 1개
xx = Dense(5, activation='relu')(input2) # dense1 layer/ input: 1, ouput:5
xx = Dense(3)(xx) # dense2 layer/ input: 5, ouput:3
xx = Dense(4)(xx) # dense3 layer/ input: 3, ouput:4
xx = Dense(6)(xx)
xx = Dense(3)(xx)
xx = Dense(7)(xx)
xx = Dense(2)(xx)
xx = Dense(3)(xx)
xx = Dense(4)(xx)
xx = Dense(10)(xx)
xx = Dense(2)(xx)
xx = Dense(7)(xx)
xx = Dense(3)(xx)
xx = Dense(12)(xx)
xx = Dense(3)(xx)
xx = Dense(7)(xx)

middle2 = Dense(3)(xx) 

# concatenate 1. 사슬같이 잇다; 연쇄시키다/ merge:병합하다.
from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2]) # merge1 == concatenate 된 layer

output1 = Dense(30)(merge1)
output1 = Dense(13)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(merge1)
output2 = Dense(32)(output2)
output2 = Dense(3)(output2)


model = Model(inputs = [input1,input2], outputs = [output1,output2]) # 어디서부터 어디까지 model이라는 걸 선언
model.summary() # 요약

"""
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
"""