# YOLO v1 _ Implementation1

아직 완성이 된  건 아니지만 일단 차근차근 설명해보고자!

## 0단계 : 라이브러리 import

```python
import tensorflow as tf
import argparse
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
%matplotlib inline
```

## 1단계 : 데이터 전처리

사용할 데이터셋은 Pascal VOC 2007 이고, 데이터셋을 다운받으면 아래와 같은 5개의 하위 폴더가 생성된다. 여기서 우리가 사용할 것은 Annotations, ImageSets, JPEGImages 폴더이다. 

![Untitled](YOLO%20v1%20_%20Implementation1%20e35d17a3233a4e92a82cda41986127ce/Untitled.png)

![Untitled](YOLO%20v1%20_%20Implementation1%20e35d17a3233a4e92a82cda41986127ce/Untitled%201.png)

Annotation 속 파일들은 xml 형태로 되어있는, 이미지 분류에 대한 정답값을 포함한 정보이다. 

예시로 가져온 annotation 데이터와 이미지파일을 비교해보면 이미지 속 객체가 car이라는 정보를 annotation 파일에서 알려줌을 확인할 수 있다.

![Untitled](YOLO%20v1%20_%20Implementation1%20e35d17a3233a4e92a82cda41986127ce/Untitled%202.png)

![Untitled](YOLO%20v1%20_%20Implementation1%20e35d17a3233a4e92a82cda41986127ce/Untitled%203.png)

xml 파일 보다 txt 파일이 다루기 더 쉬워 <object>속 데이터들을 추출해 2007 train과 2007 val 과 같은 새로운 파일을 만들어주었다. 

```python
#파서 만들기
#ArgumentParser 객체를 생성
parser = argparse.ArgumentParser(description='Build Annotations.')
#인자 추가하기
#ArgumentParser 에 프로그램 인자에 대한 정보 채우기 -> parse_args() 호출하면 default를'..'으로 하는 dir이 생성됨!
parser.add_argument('dir', default='..', help='Annotations.')

#sets 리스트
sets = [('2007', 'train'), ('2007', 'val')]

#class 넘버 지정 
classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
               'sofa': 17, 'train': 18, 'tvmonitor': 19}

#annotation 을 컨버트 해주는 함수 => xml 파일 속 object 태그에 대한 값들을 정리하는중!
def convert_annotation(year, image_id, f):
    in_file = os.path.join('yolo1\VOCdevkit\VOC%s\Annotations/%s.xml' % (year, image_id))
    #파일 가져오기
    tree = ET.parse(in_file)
    #최상위 tag 가져오기
    root = tree.getroot()

    for obj in root.iter('object'):
        #difficult 속 값 (이를 text라 부름)을 저장
        difficult = obj.find('difficult').text
        Class = obj.find('name').text
        classes = list(classes_num.keys())
        #Class가 classes 안에 없거나 difficult 의 값이 1이면 
        if Class not in classes or int(difficult) == 1:
            continue
        Class_id = classes.index(Class)
        xmlbox = obj.find('bndbox')
        #b(xmin, ymin, xmax, ymax)임!
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        #bounding box 정보와 class_id (숫자로 표시된)를 문자열로 써주기
        f.write(' ' + ','.join([str(a) for a in b]) + ',' + str(Class_id))
```

```python
#여기서! test 파일을 안 합쳐주었기에 이미지셋에서 빠져있다!
for year, image_set in sets:
    print(year, image_set)
    with open(os.path.join('yolo1\VOCdevkit\VOC%s\ImageSets\Main\%s.txt'%(year, image_set)),'r') as f:
        image_ids = f.read().strip().split()
    with open(os.path.join("yolo1\VOCdevkit", '%s_%s'%(year, image_set)),'w') as f:
        for image_id in image_ids:
            f.write('yolo1\%s\VOC%s\JPEGImages\%s.jpg' % ("VOCdevkit",year,image_id))
            convert_annotation(year, image_id, f)
            f.write('\n')
```

실행결과 :

![Untitled](YOLO%20v1%20_%20Implementation1%20e35d17a3233a4e92a82cda41986127ce/Untitled%204.png)

생성된 파일에 들어가보면 "이미지파일경로.jpg xmin, xmax, ymin, ymax, class_id" 가 텍스트 파일 형태로 저장되어 있는것을 확인할 수 있다.

![Untitled](YOLO%20v1%20_%20Implementation1%20e35d17a3233a4e92a82cda41986127ce/Untitled%205.png)

```python
import cv2 as cv
import numpy as np
```

```python
#cv2를 열심히 이용한다.
def Img_read(image_path, label):
    #이미지 불러오기
    image = cv.imread(image_path)
    #기존의 BGR 컬러를 RGB로 바꾸기
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    #이미지의 height, width 가져오고 리사이즈 해주고 
    image_h, image_w = image.shape[0:2]
    image = cv.resize(image, (448, 448))
    #여기선 픽셀범위인 0~255 를 신경망에 넣어줄 input인 0~1.0 으로 변환
    image = image / 255.

    #넘파이 배열로 (7,7,30)에 0을 다 깔아준 label_matrix
    label_matrix = np.zeros([7, 7, 30])
    for l in label:
        l = l.split(',')
        l = np.array(l, dtype=np.int)
        #각각의 좌표 얻기
        xmin = l[0]
        ymin = l[1]
        xmax = l[2]
        ymax = l[3]
        Class = l[4]
        x = (xmin + xmax) / 2 / image_w
        y = (ymin + ymax) / 2 / image_h
        w = (xmax - xmin) / image_w
        h = (ymax - ymin) / image_h
				# S=7이기에 
        loc = [7 * x, 7 * y]
        loc_i = int(loc[1])
        loc_j = int(loc[0])
        y = loc[1] - loc_i
        x = loc[0] - loc_j

        if label_matrix[loc_i, loc_j, 24] == 0:
            label_matrix[loc_i, loc_j, Class] = 1
            label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
            label_matrix[loc_i, loc_j, 24] = 1

    return image, label_matrix
```

Pascal VOC 데이터셋에 있는 이미지 데이터를 OpenCV  라이브러리를 사용해 넘파이 배열 형태의 데이터로 가공한다. (return image)

위에서 텍스트 파일로 저장한 객체 관련 데이터들을 (7,7,30) 텐서의 라벨 데이터로 만들어준다. (return label_matrix)

인풋과 아웃풋을 배치로 리턴해주는 함수를 정의한다.

- keras.utils.Sequence에 대한 자세한 정보는 [여기](https://sunshower76.github.io/frameworks/2020/02/09/Keras-Batch%EC%83%9D%EC%84%B1%ED%95%98%EA%B8%B01-(Seuquence&fit_generator)/)서 확인해볼 수 있다.

```python
from tensorflow import keras

class My_Custom_Generator(keras.utils.Sequence) :
    
    #클래스의 생성할 때 인자들을 받아옴
    def __init__(self, images, labels, batch_size) :
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
    
    #리턴해주는 길이를 범위로 해 getitem함수에 사용될 index로 반환
    def __len__(self) :
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
        #이미지개수를 배치수로 나누어 한 epoch내에 존재할 수 있는 배치수를 길이로 반환
    
    #실질적으로 배치 반환
    #함수 인자 속 index는 range(0,len)사이의 인덱스 차례대로 반환
    #해당 인덱스를 참조하여 해당인덱스에 해당하는 이미지, 마스크를 배치사이즈만큼 불러와 배치 생성
    def __getitem__(self, index) :
        batch_x = self.images[index * self.batch_size : (index+1) * self.batch_size]
        batch_y = self.labels[index * self.batch_size : (index+1) * self.batch_size]

    #train_image와 train_Label 은 리스트로 만들어주기
        train_image = []
        train_label = []

    #image_path랑 label 배치
        for i in range(0, len(batch_x)):
            img_path = batch_x[i]
            label = batch_y[i]
            image, label_matrix = Img_read(img_path, label)
            train_image.append(image)
            train_label.append(label_matrix)i
        return np.array(train_image), np.array(train_label)
```

배치 사이즈만큼 데이터들을 불러와 batch_x, batch_y를 생성하고, train_image와 train_label 안에 이들을 넣고 넘파이 배열로 반환해준다. 

```python
#모든 정보를 배열로 준비하기!

train_datasets = []
val_datasets = []

#아까 생성해준 txt 파일을 불러와서 train, val datasets안에 넣어주기
with open(os.path.join("yolo1\VOCdevkit", '2007_train'), 'r') as f:
    train_datasets = train_datasets + f.readlines()
with open(os.path.join("yolo1\VOCdevkit", '2007_val'), 'r') as f:
    val_datasets = val_datasets + f.readlines()

X_train = []
Y_train = []

X_val = []
Y_val = []

for item in train_datasets:
    item = item.replace("\n", "").split(" ")
    X_train.append(item[0])
    arr = []
    for i in range(1, len(item)):
        arr.append(item[i])
    Y_train.append(arr)

for item in val_datasets:
    item = item.replace("\n", "").split(" ")
    X_val.append(item[0])
    arr = []
    for i in range(1, len(item)):
        arr.append(item[i])
    Y_val.append(arr)
```

```python
#배치 사이즈를 정해준다
batch_size = 4

#training 에서와 validation의 batch_generator 만들어주고
my_training_batch_generator = My_Custom_Generator(X_train, Y_train, batch_size)

my_validation_batch_generator = My_Custom_Generator(X_val, Y_val, batch_size)

#__getitem__으로 배치 리턴해주기
x_train, y_train = my_training_batch_generator.__getitem__(0)
x_val, y_val = my_training_batch_generator.__getitem__(0)
print(x_train.shape)
print(y_train.shape)

print(x_val.shape)
print(y_val.shape)
```

출력 :

![Untitled](YOLO%20v1%20_%20Implementation1%20e35d17a3233a4e92a82cda41986127ce/Untitled%206.png)

아까 텍스트파일 형태로 저장해준 데이터들을 train_datasets, val_datasets에 넣고 x_train, y_train, x_val, y_val에 값을 담아준다. 이 때, x_train, x_val은 이미지 데이터라 (448,448,3)의 형태를 가지고 y_train, y_val 은 라벨 데이터이기에 (7,7,30) 형태를 가진다.

## 2단계 : 모델 구현

### 1. Pre-trained model 가져오기

```python
#yolo 모델 층 쌓기 => 지금 이 부분을 아에 pretrained 된 걸 쓰려 하는거!

nb_boxes=1
grid_w=7
grid_h=7
cell_w=64
cell_h=64
img_w=grid_w*cell_w
img_h=grid_h*cell_h
```

```python
conv_base = tf.keras.applications.densenet.DenseNet121(
    include_top=False, weights='imagenet',
    input_shape=(img_h, img_w, 3))
#네트워크 동결시키기
conv_base.trainable = False
#desnet 구조 확인하기
conv_base.summary()
```

```python
from keras import models
from keras import layers

#헤드 부분
YOLO_model = tf.keras.models.Sequential()

initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)
regularizer = tf.keras.regularizers.l2(0.0005) 

YOLO_model.add(conv_base)
YOLO_model.add(tf.keras.layers.Conv2D(1024, (3, 3), activation=lrelu, kernel_initializer=initializer, kernel_regularizer = regularizer,
 padding = 'SAME', name = "detection_conv1", dtype='float32'))
YOLO_model.add(tf.keras.layers.Conv2D(1024, (3, 3), activation=lrelu, kernel_initializer=initializer, kernel_regularizer = regularizer,
 padding = 'SAME', name = "detection_conv2", dtype='float32'))
YOLO_model.add(tf.keras.layers.MaxPool2D((2, 2)))
YOLO_model.add(tf.keras.layers.Conv2D(1024, (3, 3), activation=lrelu, kernel_initializer=initializer, kernel_regularizer = regularizer,
 padding = 'SAME', name = "detection_conv3", dtype='float32'))
YOLO_model.add(tf.keras.layers.Conv2D(1024, (3, 3), activation=lrelu, kernel_initializer=initializer, kernel_regularizer = regularizer,
 padding = 'SAME', name = "detection_conv4", dtype='float32'))
# Linear 부분
YOLO_model.add(tf.keras.layers.Flatten())
YOLO_model.add(tf.keras.layers.Dense(4096, activation=lrelu, kernel_initializer = initializer, kernel_regularizer = regularizer,
 name = "detection_linear1", dtype='float32'))
YOLO_model.add(tf.keras.layers.Dropout(.5))
# 마지막 레이어의 활성화 함수는 선형 활성화 함수인데 이건 입력값을 그대로 내보내는거라 activation을 따로 지정하지 않았다.
YOLO_model.add(tf.keras.layers.Dense(1470, kernel_initializer = initializer, kernel_regularizer = regularizer, name = "detection_linear2",
 dtype='float32')) # 7*7*30 = 1470. 0~29 : (0, 0) 위치의 픽셀에 대한 각종 출력값, 30~59 : (1, 0) 위치의...블라블라
YOLO_model.add(tf.keras.layers.Reshape((7, 7, 30), name = 'output', dtype='float32'))

YOLO_model.summary()
```

### 커스텀 학습률 스케듈러 정의하기

```python
from tensorflow import keras

class CustomLearningRateScheduler(keras.callbacks.Callback):

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

        
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # 모델의 optimizer 로 부터 현재 학습률 얻기
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # scheduled 된 학습률 얻기.
        scheduled_lr = self.schedule(epoch, lr)
        #에포크 시작하기 전 이번에 계산된 학습률을 optimizer에 설정해줌.
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))

LR_SCHEDULE = [
    # (에포크, 학습률)튜플
    (0, 0.01),
    (75, 0.001),
    (105, 0.0001),
]

#학습률 스케듈러 정의
def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr
```

### Loss Function

```python
import keras.backend as K

def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max

#IOU 값 정의: 실제 bounding box와 예측 bounding box의 합집합 면적 대 교집합 면적의 비율
def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    #교집합 면적 최소, 최대
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    #예측 bounding box
    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    #합집합 면적
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores

#yolo_head 
def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    box_wh = feats[..., 2:4] * 448

    return box_xy, box_wh

def yolo_loss(y_true, y_pred):
    label_class = y_true[..., :20]
    label_box = y_true[..., 20:24] 
    response_mask = y_true[..., 24]  
    response_mask = K.expand_dims(response_mask)  

    predict_class = y_pred[..., :20]  
    predict_trust = y_pred[..., 20:22] 
    predict_box = y_pred[..., 22:]  

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box) 
    label_xy = K.expand_dims(label_xy, 3)  
    label_wh = K.expand_dims(label_wh, 3)  
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)

    predict_xy, predict_wh = yolo_head(_predict_box) 
    predict_xy = K.expand_dims(predict_xy, 4) 
    predict_wh = K.expand_dims(predict_wh, 4)  
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)

		#iou 함수 활용
    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max) 
    best_ious = K.max(iou_scores, axis=4) 
    best_box = K.max(best_ious, axis=3, keepdims=True)  

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious)) 

		#confidence loss
    no_object_loss = 0.5 * (1 - box_mask * response_mask) * K.square(0 - predict_trust)
    object_loss = box_mask * response_mask * K.square(1 - predict_trust)
    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)

		#Classification loss
    class_loss = response_mask * K.square(label_class - predict_class)
    class_loss = K.sum(class_loss)

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  
    predict_xy, predict_wh = yolo_head(_predict_box)  

    box_mask = K.expand_dims(box_mask)
    response_mask = K.expand_dims(response_mask)
		
		#localization loss
    box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / 448)
    box_loss += 5 * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / 448)
    box_loss = K.sum(box_loss)

    loss = confidence_loss + class_loss + box_loss

    return loss
```

### weights를 저장하기 위해 콜백 추가하기

```python
# defining a function to save the weights of best model
from tensorflow.keras.callbacks import ModelCheckpoint

mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')
```

## 3단계 : 훈련하기

```python
from tensorflow import keras

YOLO_model.compile(loss=yolo_loss ,optimizer='adam')
```

```python
YOLO_model.fit(x=my_training_batch_generator,
          steps_per_epoch = int(len(train_image_dataset ) // batch_size),
          epochs = 135,
          verbose = 1,
          workers= 4,
          validation_data = my_validation_batch_generator,
          validation_steps = int(len(val_image_dataset ) // batch_size),
           callbacks=[
              CustomLearningRateScheduler(lr_schedule),
              mcp_save
          ])
```