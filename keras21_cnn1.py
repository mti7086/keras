# cnn이란 이미지 (특징을 추출한다.)

from keras.models import Sequential

filter_size = 32
kernel_size = (3,3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()
model.add(Conv2D(7, (2,2), # padding='same',
                    input_shape = (28, 28, 1)))
#model.add(Conv2D(아웃풋노드개수), (2,2)로 잘라, padding='same', 
#          input_shape = (가로, 세로,특징(1은흑백, 3은칼라))))

# model.add(Conv2D(16, (2,2)))
# model.add(MaxPooling2D(3,3))
# model.add(Conv2D(8, (2,2)))
model.add(Flatten())
# 5*5를 (2,2)로 자르면 (4,4)로 된다.
# 27*27*7(아웃풋 노드 7이 들어간 것이다.)=5103 평평하게 펴서 Dense 모델에 들어갈 수 있도록 해줌
model.add(Dense(10))
model.add(Dense(10))

model.summary()