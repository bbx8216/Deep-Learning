# 3. 다변수 선형 회귀 tensorflow 구현

> Multi-variable Linear Regression를 TensorFlow 로 구현하기

#변수 3개인 hypothesis 함수 사용 → w도 3개임

#입력 데이터를 캡해서 보여주면

![3%20%E1%84%83%E1%85%A1%E1%84%87%E1%85%A7%E1%86%AB%E1%84%89%E1%85%AE%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%BC%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%20tensorflow%20%E1%84%80%E1%85%AE%E1%84%92%E1%85%A7%E1%86%AB%209d2fe2ac71b4493db0212d9bbb394c17/Untitled.png](3%20%E1%84%83%E1%85%A1%E1%84%87%E1%85%A7%E1%86%AB%E1%84%89%E1%85%AE%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%BC%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%20tensorflow%20%E1%84%80%E1%85%AE%E1%84%92%E1%85%A7%E1%86%AB%209d2fe2ac71b4493db0212d9bbb394c17/Untitled.png)

```python
#data and label
x1 = [73.,93.,89.,96.,73.]
x2 = [80.,88.,91.,98.,66.]
x3 = [75.,93.,90.,100.,70.]
Y = [152.,185.,180.,196.,142.] #정답 혹은 레이블이라고 함

#weight
w1 = tf.Variable(10.)
w2 = tf.Variable(10.)
w3 = tf.Variable(10.)
b = tf.Variable(10.) #bias임

hypothesis = w1*x1 + w2*x2+w3*x3 + b
```

로 코드를 통해서 가설함수까지 볼 수 있다.

또 전체코드를 보면,

## 전체 Code

```python
#data and label
x1 = [73.,93.,89.,96.,73.]
x2 = [80.,88.,91.,98.,66.]
x3 = [75.,93.,90.,100.,70.]
Y = [152.,185.,180.,196.,142.] #정답 혹은 레이블이라고 함

# random weights
w1 = tf.Variable(tf.random_normal([1]))
w2 = tf.Variable(tf.random_normal([1]))
w3 = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

learning_rate = 0.000001

#gradient descent 통해 w update
for i in range(1000+1):
	with tf.GradientTape as tape:
		hypothesis = w1*x1 + w2*x2 +w3*x3 +b
		cost = tf.reduce_mean(tf.square(hypothesis - Y))

	w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1,w2,w3,b])

	#update w1, w2, w3 and b
	w1.assign_sub(learning_rate *w1_grad)
	w2.assign_sub(learning_rate *w2_grad)
	w3.assign_sub(learning_rate *w3_grad)
	b.assign_sub(learning_rate *b_grad)

	if i % 50 == 0:
		print("{:5}|{:12.4f}".format(i, cost.nnumpy()))

```

매트릭스 사용하면 더 간결하게 표현할 수 있음

### 데이터를 매트릭스 이용해 확인

```python
data = np.array({
	#x1, x2, x3, y
	[73.,80.,75.,152.],
	[93.,88.,93.,185.],
	[89.,91.,90.,180.],
	[96.,98.,100.,196.],
	[73.,66.,70.,142.]
], dtype = np.float32)

#slice data , numpy 통해서 하는 거
X = data[:,:-1]#[모든 로우, 마지막을 뺀 칼럼]
y = data[:,[-1]]#[모든 로우, 마지막 칼럼]

w = tf.Variable(tf.random_normal([3,1]))
b = tf.Variable(tf.random_normal([1]))

#hypothesis, prediction function
def predict(x):
	return tf.matmul(X,W)+b
```

## Matrix 반영한 전체 Code

```python
data = np.array({
	#x1, x2, x3, y
	[73.,80.,75.,152.],
	[93.,88.,93.,185.],
	[89.,91.,90.,180.],
	[96.,98.,100.,196.],
	[73.,66.,70.,142.]
], dtype = np.float32)

#slice data , numpy 통해서 하는 거
X = data[:,:-1]#[모든 로우, 마지막을 뺀 칼럼]
y = data[:,[-1]]#[모든 로우, 마지막 칼럼]

w = tf.Variable(tf.random_normal([3,1]))
b = tf.Variable(tf.random_normal([1]))

learning_rate = 0.000001

#hypothesis, prediction function
def predict(x):
	return tf.matmul(X,W)+b

n_epochs = 2000
for i in range(n_epochs+1):
	#record the gradient of the cost function
	with tf.GradientTape() as tape:
		cost = tf.reduce_mean((tf.square(predict(X) - y)))

	#calculates the gradients of the loss
	W_grad, b_grad = tape.gradient(cost, [W,b])

	#updates parameters (W and b)
	W.assign_sub(learning_rate * W_grad)
	b.assign_sub(learning_rate * b_grad)

	if i%100 == 0:
		print("{:5}|{:10.4f}".format(i, cost.numpy())
```

전체 code 와 매트릭스를 사용한 전체 code를 비교해보면 

![3%20%E1%84%83%E1%85%A1%E1%84%87%E1%85%A7%E1%86%AB%E1%84%89%E1%85%AE%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%BC%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%20tensorflow%20%E1%84%80%E1%85%AE%E1%84%92%E1%85%A7%E1%86%AB%209d2fe2ac71b4493db0212d9bbb394c17/Untitled%201.png](3%20%E1%84%83%E1%85%A1%E1%84%87%E1%85%A7%E1%86%AB%E1%84%89%E1%85%AE%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%92%E1%85%A7%E1%86%BC%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%20tensorflow%20%E1%84%80%E1%85%AE%E1%84%92%E1%85%A7%E1%86%AB%209d2fe2ac71b4493db0212d9bbb394c17/Untitled%201.png)

다변수 쓸 땐 매트릭스를 사용하는게 아주 편함!