# 4. 로지스틱 회귀/ 분류를 TensorFlow 로 구현

1. 데이터 scatter 로 확인

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#아래 데이터에서 1. 이렇게 점을 붙인 이유 : 실수형태의 자료형-> 그래야 나중에 오류가 안남
x_train = [[1.,2.],[2.,3.],[3.,1.],[4.,3.],[5.,3.],[6.,2.]]
y_train = [[0.],[0.],[0.],[1.],[1.],[1.]]

x_test = [[5.,2.]]
y_test = [[1.]]

x1 = [x[0] for x in x_train] #1,2,3,...
x2 = [x[1] for x in x_train] #2,3,1...

colors = [int(y[0]%3) for y in y_train] #colors를 먼저 정의해 주고
plt.scatter(x1, x2, c=colors, marker='^') #이렇게 그래프 그리기
plt.scatter(x_test[0][0], x_test[0][1], c="red")

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
```

실행해보면 아래와 같이 그래프가 그려지고

![4%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%85%E1%85%B3%E1%86%AF%20TensorFlow%20%E1%84%85%E1%85%A9%20%E1%84%80%E1%85%AE%E1%84%92%E1%85%A7%E1%86%AB%20d9c7774a0bdd4415a9ad64ff90c4664a/Untitled.png](4%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%85%E1%85%B3%E1%86%AF%20TensorFlow%20%E1%84%85%E1%85%A9%20%E1%84%80%E1%85%AE%E1%84%92%E1%85%A7%E1%86%AB%20d9c7774a0bdd4415a9ad64ff90c4664a/Untitled.png)

```python
#텐서플로의 데이터api 이용해 데이터 텐서 슬라이스 해주기
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

W = tf.Variable(tf.zeros([2,1]), name = 'weight') #2행 1열ㄱ
b = tf.Variable(tf.zeros([1]), name = 'bias')

#로지스틱 회귀 함수 선언 => 가설 함수 반환
def logistic_regression(features):
    hypothesis = tf.divide(1., 1. + tf.exp(-tf.matmul(features, W)+b)) #시그모이드 함수 표현한 것.
    return hypothesis

#손실함수 선언, 수식을 그대로 넣는거임 (수식은 그림 참고)
def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(labels*tf.math.log(hypothesis)+(1-labels)*tf.math.log(1-hypothesis))
    return cost

#옵티마이저 선언, 학습률 선언해 각각의 코스트 값을 줄이는... (추가 검색 필요)
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)

#정확도 함수 선언 
def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis>0.5,dtype=tf.float32) # hypothesis값이 0.5 보다 크면 1의 값, 작으면 0값 반환
    accuracy = tf.reduce_mean(tf.equal(predicted, labels),tf.int32)#label 값이 0,1을 가지기에 predicted 값과 비교해서 정확도 
    return accuracy
```

```python
#gradient 함수 선언
def grad(features, labels):
    with tf.GradientTape() as tape: #각각의 미분값 기록
        hypothesis = logistic_regression(features)
        loss_value = loss_fn(hypothesis, labels)
    return tape.gradient(loss_value, [W,b]) #미분값 계산해서 반환하는 함수를 통한 반환

#1000회 학습
EPOCHS = 1001
for step in range(EPOCHS):
    for features, labels in iter(dataset.batch(len(x_train))):
        hypothesis = logistic_regression(features)
        grads = grad(features, labels)
        optimizer.apply_gradients(grads_and_vars = zip(grads,[W,b]))
        if step % 100 ==0:
            print("Iter: {}, Loss : {:.4f}".format(step,loss_fn(hypothesis,labels)))
            
test_acc = accuracy_fn(logistic_regression(x_test),y_test)
print("Test Result = {}".format(tf.cast(logistic_regression(x_test)>0.5, dtype=tf.int32)))
print("Testset Accuracy: {:.4f}".format(test_acc))
```

다 실행해주면 

![4%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%85%E1%85%B3%E1%86%AF%20TensorFlow%20%E1%84%85%E1%85%A9%20%E1%84%80%E1%85%AE%E1%84%92%E1%85%A7%E1%86%AB%20d9c7774a0bdd4415a9ad64ff90c4664a/Untitled%201.png](4%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%20%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%E1%84%85%E1%85%B3%E1%86%AF%20TensorFlow%20%E1%84%85%E1%85%A9%20%E1%84%80%E1%85%AE%E1%84%92%E1%85%A7%E1%86%AB%20d9c7774a0bdd4415a9ad64ff90c4664a/Untitled%201.png)

이건 잘 뜨는데 그 다음 문장부턴 오류가 발생한다..

해결해려보려 했는데 일단 안햇음.. ㅎㅋㅋ왜냐면 하다가 스터디 해서,,,