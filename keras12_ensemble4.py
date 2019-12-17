# 이건 숙제로 직접해본 것입니다.
# import keras # keras 가져오겠다.
# import tensorflow

"""
#1. 데이터
import numpy as np # numpy를 가져와 np로 지정해주겠다.

x1 = np.array([range(100), range(311,411), range(100)])
x2 = np.array([range(501, 601), range(711,811), range(100)])

y1 = np.array([range(101, 201), range(311,411), range(100, 200)])
y2 = np.array([range(501, 601), range(711,811), range(100)])
y3 = np.array([range(401, 501), range(211,311), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

print(x1.shape) # (100,3)
print(x2.shape) # (100,3)
print(y1.shape) # (100,3)
print(y2.shape) # (100,3)
print(y3.shape) # (100,3)

x1x2 = np.append(x1, x2, axis=1)
y1y2 = np.append(y1, y2, axis=1)
print(x1x2.shape)
print(y1y2.shape)

from sklearn.model_selection import train_test_split
x1x2_train, x1x2_test, y1y2_train, y1y2_test = train_test_split(
    x1x2, y1y2, random_state=33, test_size=0.4, shuffle=False
) # x 값을 x_train, x_test로 분류 y 값을 y_train, y_test로 분류
x1x2_val, x1x2_test, y1y2_val, y1y2_test = train_test_split(
    x1x2_test, y1y2_test, random_state=33, test_size=0.5, shuffle=False # val 20% test 20% train 60%
) # x_test 값을 x_val, x_test로 분류 y_test 값을 y_val, y_test로 분류
# test를 40% 주겠다. train을 60%주겠다
y3_train, y3_test = train_test_split(
    y3, random_state=33, test_size=0.4, shuffle=False
)
y3_val, y3_test = train_test_split(
    y3_test, random_state=33, test_size=0.5, shuffle=False
)

print(x1x2_test.shape) #(20,3)
# random_state란

# random_state=50 순서는 50개 골라서 바뀌지만 데이터 분석하는데 문제 없다.
# 고정 된 random_state 일 때, 프로그램 실행마다 똑같은 결과를 산출합니다
# random_state을 설정하지 않으면 알고리즘을 실행할 때마다 다른 시드가 사용되며 다른 결과가 나옵니다. 
# x = (1, 2, 3, 4, 5) random_state하면 -> x = (1, 3, 4, 2, 5) 즉 순서는 바껴도 1이 1나오는 결과는 똑같다.
# y = (1, 2, 3, 4, 5) random_state하면 -> x = (1, 3, 4, 2, 5) 즉 순서는 바껴도 1이 1나오는 결과는 똑같다.



#2. 모델구성(함수형 모델)
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential() # 순차적모델

input1 = Input(shape=(6, )) # 최초의 input/ 컬럼 = 열 = 1개
dense1 = Dense(5, activation='relu')(input1) # dense1 layer/ input: 1, ouput:5
dense2 = Dense(3)(dense1) # dense2 layer/ input: 5, ouput:3
dense3 = Dense(4)(dense2) # dense3 layer/ input: 3, ouput:4
dense4 = Dense(6)(dense3)
dense5 = Dense(3)(dense4)

middle1 = Dense(6)(dense5)

# concatenate 1. 사슬같이 잇다; 연쇄시키다/ merge:병합하다.
from keras.layers.merge import concatenate
output1 = Dense(30)(middle1)
output1 = Dense(13)(output1)
output1 = Dense(6)(output1)

output2 = Dense(15)(middle1)
output2 = Dense(32)(output2)
output2 = Dense(3)(output2)

model = Model(inputs = input1, 
              outputs = [output1,output2]) # 어디서부터 어디까지 model이라는 걸 선언
model.summary() # 요약


#3. 훈련
model.compile(loss='mse', optimizer='adam', # metrics=['accuracy']) # loss=손실함수 optimizer=최적화
                metrics=['mse']) # loss: 6.8923e-13, metrics->mse: 6.8923e-13
# / 이 모듈에서 최소한으로 손실보겠다. accuracy 정확성
# model.fit(x_train, y_train, epochs=500, batch_size=1) # fit == 트레이닝 100번 훈련 1,2,3,4,5를 한개씩(1) 잘라서 훈련
model.fit(x1x2_train,[y1y2_train, y3_train], epochs=100, batch_size=1,
        validation_data=(x1x2_val,[y1y2_val, y3_val])) # validation_data=머신한테 니가 검증해가면서 학습하라는 뜻
# batch_size란 sample데이터 중 한번에 네트워크에 넘겨주는 데이터의 수를 말한다.
# 1:1 과외가 공부 잘되는 것 처럼 1억개 데이터이면 100만개씩 모아서 공부시키듯이


#4. 평가 예측
# mse = 평균 제곱 에러 오답에 가까울수록 큰 값이 나온다. 반대로 정답에 가까울수록 작은 값이 나온다.
mse = model.evaluate(x1x2_test, [y1y2_test, y3_test], batch_size=1) # evaluate함수에 x_test, y_test값을 넣으면 loss a[0], acc a[1] 값이 나온다.
print("mse : ", mse[0])
print("mse : ", mse[1])
print("mse : ", mse[2])
print("mse : ", mse[3])
print("mse : ", mse[4])

y1_predict, y2_predict = model.predict(x1x2_test)
print(y1_predict, y2_predict)


# RMSE 구하기 RMSE = MSE에 루트를 씌워준 것
from sklearn.metrics import mean_squared_error
def RMSE(xxx, yyy):
    return np.sqrt(mean_squared_error(xxx, yyy)) # mse = mean squared error/sqrt=루트값
    # y_predict와 주어진 값 y_test와 비교
RMSE1 = RMSE(y1y2_test, y1_predict)
RMSE2 = RMSE(y3_test, y2_predict)

print("RMSE1: ", RMSE1)
print("RMSE2: ", RMSE2)
print("RMSE: ", (RMSE1+RMSE2)/2)


# R2 구하기 R2는 1이 나오면 잘한거 0이 나오면 최악이라는 지표일 뿐 0.9999 잘 됬다는 것은 아니다.
from sklearn.metrics import r2_score
r2_y1y2_predict = r2_score(y1y2_test, y1_predict)
r2_y3_predict = r2_score(y3_test, y2_predict)

print("R2_y1y2 : ", r2_y1y2_predict)
print("R2_y3 : ", r2_y3_predict)
print("R2 : ", (r2_y1y2_predict + r2_y3_predict)/2)

# R2 0.9999999999988425 RMSE 3.0902579776801397e-06
# R2의 값을 늘렸더니 R2 0.9999999999992074 늘고 , RMSE의 값은 낮아졌다. RMSE 3.846355047163435e-06
"""

# 이것은 교수님께서 하신 코드입니다.
# import keras # keras 가져오겠다.
# import tensorflow

#1. 데이터
import numpy as np # numpy를 가져와 np로 지정해주겠다.

x1 = np.array([range(100), range(311,411), range(100)])
# x2 = np.array([range(501, 601), range(711,811), range(100)])

y1 = np.array([range(101, 201), range(311,411), range(100, 200)])
y2 = np.array([range(501, 601), range(711,811), range(100)])
# y3 = np.array([range(401, 501), range(211,311), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
# x2 = np.transpose(x2)
y2 = np.transpose(y2)
# y3 = np.transpose(y3)

# print(x1.shape) # (100,3)
# print(x2.shape) # (100,3)
# print(y1.shape) # (100,3)
# print(y2.shape) # (100,3)
# print(y3.shape) # (100,3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=33, test_size=0.4, shuffle=False
) # x 값을 x_train, x_test로 분류 y 값을 y_train, y_test로 분류
x1_val, x1_test, y1_val, y1_test = train_test_split(
    x1_test, y1_test, random_state=33, test_size=0.5, shuffle=False # val 20% test 20% train 60%
) # x_test 값을 x_val, x_test로 분류 y_test 값을 y_val, y_test로 분류
# test를 40% 주겠다. train을 60%주겠다
y2_train, y2_test = train_test_split(
    y2, random_state=33, test_size=0.4, shuffle=False
)
y2_val, y2_test = train_test_split(
    y2_test, random_state=33, test_size=0.5, shuffle=False # val 20% test 20% train 60%
)
# x2_train, x2_test = train_test_split(
#     x2, random_state=33, test_size=0.4, shuffle=False
# )
# x2_val, x2_test = train_test_split(
#     x2_test, random_state=33, test_size=0.5, shuffle=False
# )

# print(x2_test.shape) #(20,3)
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

middle1 = Dense(3)(dense5)

# input2 = Input(shape=(3, )) # 최초의 input/ 컬럼 = 열 = 1개
# xx = Dense(5, activation='relu')(input2) # dense1 layer/ input: 1, ouput:5
# xx = Dense(3)(xx) # dense2 layer/ input: 5, ouput:3
# xx = Dense(4)(xx) # dense3 layer/ input: 3, ouput:4
# xx = Dense(6)(xx)
# xx = Dense(3)(xx)

# middle2 = Dense(3)(xx) 

# concatenate 1. 사슬같이 잇다; 연쇄시키다/ merge:병합하다.
from keras.layers.merge import concatenate
# merge1 = concatenate([middle1, middle2]) # merge1 == concatenate 된 layer

output1 = Dense(30)(middle1)
output1 = Dense(13)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(middle1)
output2 = Dense(32)(output2)
output2 = Dense(3)(output2)

# output3 = Dense(30)(merge1)
# output3 = Dense(16)(output3)
# output3 = Dense(3)(output3)

# model = Model(inputs = [input1,input2], 
#               outputs = [output1,output2,output3]) # 어디서부터 어디까지 model이라는 걸 선언
model = Model(inputs = input1, 
              outputs = [output1, output2])
                         
model.summary() # 요약


#3. 훈련
model.compile(loss='mse', optimizer='adam', # metrics=['accuracy']) # loss=손실함수 optimizer=최적화
                metrics=['mse']) # loss: 6.8923e-13, metrics->mse: 6.8923e-13
# / 이 모듈에서 최소한으로 손실보겠다. accuracy 정확성
# model.fit(x_train, y_train, epochs=500, batch_size=1) # fit == 트레이닝 100번 훈련 1,2,3,4,5를 한개씩(1) 잘라서 훈련
model.fit(x1_train,[y1_train,y2_train], epochs=100, batch_size=1,
        validation_data=(x1_val,[y1_val,y2_val])) # validation_data=머신한테 니가 검증해가면서 학습하라는 뜻
# batch_size란 sample데이터 중 한번에 네트워크에 넘겨주는 데이터의 수를 말한다.
# 1:1 과외가 공부 잘되는 것 처럼 1억개 데이터이면 100만개씩 모아서 공부시키듯이



#4. 평가 예측
# mse = 평균 제곱 에러 오답에 가까울수록 큰 값이 나온다. 반대로 정답에 가까울수록 작은 값이 나온다.
mse = model.evaluate(x1_test, [y1_test,y2_test], batch_size=1) # evaluate함수에 x_test, y_test값을 넣으면 loss a[0], acc a[1] 값이 나온다.
print("mse : ", mse[0])
print("mse : ", mse[1])


y1_predict, y2_predict = model.predict(x1_test)
print(y1_predict, y2_predict)

"""
# RMSE 구하기 RMSE = MSE에 루트를 씌워준 것
from sklearn.metrics import mean_squared_error
def RMSE(xxx, yyy):
    return np.sqrt(mean_squared_error(xxx, yyy)) # mse = mean squared error/sqrt=루트값
    # y_predict와 주어진 값 y_test와 비교
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE3 = RMSE(y3_test, y3_predict)

print("RMSE1: ", RMSE1)
print("RMSE2: ", RMSE2)
print("RMSE3: ", RMSE3)
print("RMSE: ", (RMSE1+RMSE2+RMSE3)/3)

# R2 구하기 R2는 1이 나오면 잘한거 0이 나오면 최악이라는 지표일 뿐 0.9999 잘 됬다는 것은 아니다.
from sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, y1_predict)
r2_y2_predict = r2_score(y2_test, y2_predict)
r2_y3_predict = r2_score(y3_test, y3_predict)

print("R2_y1 : ", r2_y1_predict)
print("R2_y2 : ", r2_y2_predict)
print("R2_y3 : ", r2_y3_predict)
print("R2 : ", (r2_y1_predict + r2_y2_predict + r2_y3_predict)/3)

# R2 0.9999999999988425 RMSE 3.0902579776801397e-06
# R2의 값을 늘렸더니 R2 0.9999999999992074 늘고 , RMSE의 값은 낮아졌다. RMSE 3.846355047163435e-06
"""