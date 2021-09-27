# 1. Simple Linear Regression 를 TensorFlow로 구현하기

## 1. Hypothesis

Simple Linear Regression 이기에 가설함수 H(x)는 : 

H(x) = Wx+b

## 2. Cost function 함수 구현

cost(W,b) = 차이제곱의 평균 함수! 

```python
cost = tf.reduce_mean(tf.square(hypothesis - y_data))
```

cost function 최소화 알고리즘

→ 경사하강법

### Gradient descent (경사하강법) 구현

```python
#learning_rate initialize
learning_rate = 0.01

#Gradient descent를 텐서플로에선 GradientTape을 이용해서 구현
#with 구문안에 있는 이 블록안의 변수들의 변화를 테이프에 기록하기
#이 tpae 안의 변수는 W, b 가 있음.
with tf.GradientTape() as tape:
	hypothesis = W * x_data + b
	cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#gradient 함수는 이 함수에 대해서 변수들에 대한 개별 미분 값(기울기 값)을 구해 튜플로 반환 
W_grad, b_grad = tape.gradient(cost, [W,b])

#assign_sub는 뺀 값을 다시 그 값에 할당 (즉 "-="과 같음)
W.assign_sub(learning_rate *W_grad)
b.assgin_sub(learning_rate * b_grad)
```

### 파라미터 (W,b) 업데이트

```python
W = tf.Variable(2.9)
b= tf.Variable (0.5)

for i in range (100):
	#Gradient descent
	#방금 위에서 구현해준 부분의 코드를 그대로
	#결국 경사하강 백번 수행해서 비용함수의 min 값 찾아내고 싶은 것
	
	# W,b,cost의 중간 중간 값이 어떻게 변화는지 확인하고 싶어서
	if i%10==0:
		print("{:5}|{:10.4}|{:10.6f}".format(i,W.numpy(), b.numpy(), cost))
```

## Full Code

```python
import tensorflow as tf
tf.enable_eager_execution()

#Data
x_data = [1,2,3,4,5]
y_data = [1,2,3,4,5]

#W,b initialize
W = tf.Variable(2.9)
b= tf.Variable (0.5)

#learning_rate initialize
learning_rate = 0.01

#W,b update
for i in range (100+1): #항상 궁금한건데, 왜 굳이 100+1 이런식으로 표기하는 지 모르겠다.
	#Gradient descent
	with tf.GradientTape() as tape:
		hypothesis = W * x_data + b
		cost = tf.reduce_mean(tf.square(hypothesis - y_data))
	W_grad, b_grad = tape.gradient(cost, [W,b])
	W.assign_sub(learning_rate *W_grad)
	b.assgin_sub(learning_rate * b_grad)
	if i%10==0:
		print("{:5}|{:10.4}|{:10.6f}".format(i,W.numpy(), b.numpy(), cost))
```

 

결과를 통해 에포크가 늘어날수록 우리의 가설함수와 테스트 데이터가 가까워짐을 확인할 수 있다.