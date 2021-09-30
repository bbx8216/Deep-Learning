# 1. 데이터 전처리

PASCAL VOC 2007 에서 얻어내야할 정보

1. x_train , x_test, y_train, y_test 에 데이터를 넣어준다.

```python
x_train = '\yolo1\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages'
y_train = '\yolo1\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\Annotations'

x_test = '\yolo1\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages'
y_test = '\yolo1\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\Annotations'
```

1. annotation 은 xml 파일로 제공된다. 

→ 여기서 고민인게 xml 파일을 그대로 가져다 써도 괜찮고 text 파일로 변환해서 사용해도 되는데, 나한텐 둘 ㅏㄷ 어렵다는 점이다.. 그래도 확실히 민규님 코드가 더 잘이해된다..

#민규님 코드에서 get_Classes_inImage(xml_file_list)에서 결국 반환하는것은 Classes_inDataSet 이다. 그래서 난 이걸 그냥 알려주기로 했다.

```python
classes
```

xml 파일을 text 파일로 변환하고 cnn(코드로 제공된 - 마찬가지로 무슨 모델인지 모르겠다,, 그냥 yolo cnn 그대로 만든거인듯!)

전처리를 다 하고 나면 모델을 훈련시켜야 한다. 이 때 class 를 통해 custom generator 클래스를 만들어 쓴다.

이 때 합성곱 신경망 공부할 때 배운 내용이 쓰인다! 

```python
#label 획득 부분
def get_label_fromImage(xml_file_path, Classes_inDataSet):

    f = open(xml_file_path)
    xml_file = xmltodict.parse(f.read()) 

    Image_Height = float(xml_file['annotation']['size']['height'])
    Image_Width  = float(xml_file['annotation']['size']['width'])

    label = np.zeros((7, 7, 25), dtype = float)
    
    try:
        for obj in xml_file['annotation']['object']:
            
            # class의 index 휙득
            class_index = Classes_inDataSet.index(obj['name'].lower())
            
            # min, max좌표 얻기
            x_min = float(obj['bndbox']['xmin']) 
            y_min = float(obj['bndbox']['ymin'])
            x_max = float(obj['bndbox']['xmax']) 
            y_max = float(obj['bndbox']['ymax'])

            # 224*224에 맞게 변형시켜줌
            x_min = float((224.0/Image_Width)*x_min)
            y_min = float((224.0/Image_Height)*y_min)
            x_max = float((224.0/Image_Width)*x_max)
            y_max = float((224.0/Image_Height)*y_max)

            # 변형시킨걸 x,y,w,h로 만들기 
            x = (x_min + x_max)/2.0
            y = (y_min + y_max)/2.0
            w = x_max - x_min
            h = y_max - y_min

            # x,y가 속한 cell알아내기
            x_cell = int(x/32) # 0~6
            y_cell = int(y/32) # 0~6
            # cell의 중심 좌표는 (0.5, 0.5)다
            x_val_inCell = float((x - x_cell * 32.0)/32.0) # 0.0 ~ 1.0
            y_val_inCell = float((y - y_cell * 32.0)/32.0) # 0.0 ~ 1.0

            # w, h 를 0~1 사이의 값으로 만들기
            w = w / 224.0
            h = h / 224.0

            class_index_inCell = class_index + 5

            label[y_cell][x_cell][0] = x_val_inCell
            label[y_cell][x_cell][1] = y_val_inCell
            label[y_cell][x_cell][2] = w
            label[y_cell][x_cell][3] = h
            label[y_cell][x_cell][4] = 1.0
            label[y_cell][x_cell][class_index_inCell] = 1.0

    # single-object in image
    except TypeError as e : 
        # class의 index 휙득
        class_index = Classes_inDataSet.index(xml_file['annotation']['object']['name'].lower())
            
        # min, max좌표 얻기
        x_min = float(xml_file['annotation']['object']['bndbox']['xmin']) 
        y_min = float(xml_file['annotation']['object']['bndbox']['ymin'])
        x_max = float(xml_file['annotation']['object']['bndbox']['xmax']) 
        y_max = float(xml_file['annotation']['object']['bndbox']['ymax'])

        # 224*224에 맞게 변형시켜줌
        x_min = float((224.0/Image_Width)*x_min)
        y_min = float((224.0/Image_Height)*y_min)
        x_max = float((224.0/Image_Width)*x_max)
        y_max = float((224.0/Image_Height)*y_max)

        # 변형시킨걸 x,y,w,h로 만들기 
        x = (x_min + x_max)/2.0
        y = (y_min + y_max)/2.0
        w = x_max - x_min
        h = y_max - y_min

        # x,y가 속한 cell알아내기
        x_cell = int(x/32) # 0~6
        y_cell = int(y/32) # 0~6
        x_val_inCell = float((x - x_cell * 32.0)/32.0) # 0.0 ~ 1.0
        y_val_inCell = float((y - y_cell * 32.0)/32.0) # 0.0 ~ 1.0

        # w, h 를 0~1 사이의 값으로 만들기
        w = w / 224.0
        h = h / 224.0

        class_index_inCell = class_index + 5

        label[y_cell][x_cell][0] = x_val_inCell
        label[y_cell][x_cell][1] = y_val_inCell
        label[y_cell][x_cell][2] = w
        label[y_cell][x_cell][3] = h
        label[y_cell][x_cell][4] = 1.0
        label[y_cell][x_cell][class_index_inCell] = 1.0

    return label # np array로 반환
```