# cnn(Convolution Neural Network)이란 이미지 (특징을 추출한다.)

from keras.models import Sequential

filter_size = 32
kernel_size = (3,3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()
model.add(Conv2D(3, (3,3), padding='same',
                    input_shape = (28, 28, 1))) # (None, 27,27,7) -> 3차원
# padding의 default값은 valid/ padding='same'은 원래 기존 5,5랑 같아짐
# padding='same'을 하는 이유는 데이터 손실치를 조금이라도 줄이기 위해서 사용한다.

model.add(Conv2D(4, (2,2))) # 위의 model과 같은 모양이니 한번더 쓸 수 있음

#model.add(Conv2D(아웃풋노드개수,filter), (2,2)로 잘라, padding='same', 
#          input_shape = (가로, 세로,특징(1은흑백, 3은칼라))))
# padding ='same' 우리 원래 있던 이미지에 (2,2)로 잘라 (27,27)로 된 것이
# 원래 있던 이미지에 가로,세로로 한칸씩 더 씌워 (2,2)로 잘라도 (28,28)로 만들어준다.

model.add(Conv2D(16, (2,2)))
# model.add(MaxPooling2D(3,3))
model.add(Conv2D(8, (2,2)))
model.add(Flatten()) # (None, 5103) -> 2차원
# 5*5를 (2,2)로 자르면 (4,4)로 된다.
# 27*27*7(아웃풋 노드 7이 들어간 것이다.)=5103 평평하게 펴서 Dense 모델에 들어갈 수 있도록 해줌
model.add(Dense(10)) # (None, 10)
model.add(Dense(1)) 

model.summary()