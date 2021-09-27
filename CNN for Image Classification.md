
# 이미지 분류를 위한 합성곱 신경망 공부

[np.dot](http://np.dot) vs tf.matmul ? 

둘다 행렬 곱 아냐?

대부분 딥러닝 패키지들은 합성곱 신경망 만들 때 합성곱이 아니라 교차 상관을 사용

→ 싸이파이의 correlate() 함수 사용하면 간단히 계산할 수 있음.

패딩 : 원본 배열의 양 끝에 빈 원소를 추가하는 것

스트라이드 : 미끄러지는 배열의 간격을 조절하는 것

풀 패딩 : 원본 배열 원소의 연산 참여도를 동일하게 만든다.

correlate() 함수에 풀패딩 적용하려면 매개변수 mode 를 full로 지정

```python
correlate(x,w,mode = 'full')
```

세임 패딩 : 출력 배열의 길이를 원본 배열의 길이와 동일하게 만듦

mode = 'same'

스트라이드 : 미끄러지는 간격 조절

싸이파이의 correlate2d() 함수 이용해 2차원 배열의 합성곱 계산

```python
x=np.array([[1,2,3],
           [4,5,6],
           [7,8,9]])
w = np.array([[2,0],[0,0]])
from scipy.signal import correlate2d
correlate2d(x,w,mode='valid')
```

### 텐서플로로 합성곱 수행

원본배열: 입력, 미끄러지는 배열 : 가중치 라 부른다.

합성곱 신경망의 입력은 일반적으로 4차원 배열

2차원 합성곱 수행함수 : conv2d() : 이 함수는 입력으로 4차원 배열을 기대. 

입력 구조 : (배치, 샘플의 높이, 샘플의 너비, 컬러 채널의 차원)

가중치 구조 : (가중치의 높이, 가중치의 너비, 채널, 가중치 개수)

```python
import tensorflow as tf
x_4d = x.astype(np.float).reshape(1,3,3,1)
w_4d = w.reshape(2,2,1,1)
c_out = tf.nn.conv2d(x_4d, w_4d, strides=1, padding ='SAME')
```

conv2d() 함수는 결괏값으로 tensor 객체를 반환

다차원 배열 : 텐서

tensor 객체의 numpy() 메서드를 사용하면 텐서를 넘파이 배열로 변환할 수 있음.

풀링연산

입력이 합성곱층을 통과할 때 합성곱과 활성화 함수가 적용되어 특성 맵이 만들어진다.

그런 다음 특성 맵이 풀링층을 통과하여 또 다른 특성 맵이 만들어진다.

합성곱층 뒤에 풀링층이 뒤따르는 형태는 합성곱 신경망의 전형적인 모습이다.

풀링 : 특성 맵을 스캔하며 최댓값을 고르거나 평균값을 계산하는 것을 말한다. 

최대 풀링 : 최댓값 고르는 방식 ⇒ 맵 크기 줄임

합성곱층에서 승트라이드를 크게 지정하여 특성 맵의 크기를 줄이면 안되나? 

경험적으로 합성곱층에 세임 패딩을 적용하고 풀링층에서 특성 맵의 크기를 줄이는 것이 더 효과적

max_pool2d() 함수를 사용하면 최대풀링 수행

매개변수 값으로 풀링 크기와 스트라이드만 전달하면 자동으로 최대 풀링 수행해 입력값 반으로 줄여줌. 

풀링의 크기를 ksize 매개변수에, 스트라이드 크기를 strides 매개변수에 동일한 값으로 지정.

max_pool2d() 반환한 tensor 객체를 numpy() 메서드로 변환한 다음 2X2 크기의 2차원 배열로 변형해보면

```python
p_out = tf.nn.max_pool2d(x, ksize=2, strides = 2, padding = 'VALID')
p_out.numpy().reshape(2,2)
```

풀링층에는 학습되는 가중치가 없다.

즉, 풀링층 통과 전후로 배치 크기와 채널 크기는 동일하다. 

합성곱 신경망의 구조를 알아본다.

은닉층에 시그모이드 함수를 활성화 함수로 사용했었다. 출력층은 이진분류 : 시그모이드, 다중 분류 : 소프트맥스 함수 사용.

렐루 함수 : 주로 합성곱층에 적용되는 활성화 함수, 합성곱 신경망의 성능을 더 높여줌.

구현 : 넘파이의 maximum() 함수를 사용하면 간단하게 구현.

```python
def relu(x):
	return np.maximum(x,0)
```

텐서플로가 제공하는 렐루 함수는 relu()

텐서플로의 렐루 함수는 tensor 객체를 반환하므로 화면에 출력하려면 넘파이로 변환해야.

```python
r_out = tf.nn.relu(x)
r_out.numpy()
```

합성곱 신경망에 주입될 입력 데이터에는 채널이 있다.

이미지는 채널이라는 차원을 하나 더 가진다. 

채널이란 이미지의 픽셀이 가진 색상을 표현하기 위해 필요한 정보

이미지의 모든 채널에 합성곱이 한 번에 적용되어야 하므로 커널의 마지막 차원은 입력 채널의 개수와 동일해야 한다.

이미지에서 여러 개의 특징을 감지하려면 복수 개의 커널을 사용해야함.

풀링층에서 일어나는 일

합성곱층을 통해 특성 맵이 만들어 졌고, 이 특성 맵에 활성화 함수로 렐루 함수를 적용하고 풀링을 적용한다. 

특성 맵을 펼쳐 완전 연결 신경망에 주입.

특성 맵은 일렬로 펼쳐 완전 연결층에 입력으로 주입한다. 합성곱층과 풀링층을 거쳐 만들어진 특성 맵이 완전 연결층에 주입됨.

완전 연결층의 출력은출력층의 뉴런과 연결됨.

합성곱 신경망을 만들고 훈련.

텐서플로가 제공하는 ㅎ바성곱 함수와 자동 미분 기능을 사용

합성곱층 → 활성화 함수 → 풀링층 → (펼침) → 완전 연결층 →(소프트맥스 함수) → 출력

### 합성곱 신경망의 정방향 계산 구현

forpass() 메서드를 구현한 뒤 앞에 합성곱과 풀링층 추가한다.

1. 합성곱 적용하기

```python
def forpass(self, x):
	#3X3 합성곱 연산을 수행
	c_out = tf.nn.conv2d(x,self.conv_w, strides = 1, padding = 'SAME') + self.conv_b
```

1) self.conv_w 

: self.con_w 는 합성곱에 사용할 가중치임. 3*3*1 크기의 커널을 10개 사용하므로 가중치의 전체 크기는 3*3*1*10dla

2) strides, padding

특성 맵의 가로와 세로 크기를 일정하게 만들기 위하여 strides = 1, padding = 'SAME'으로 지정

1. 렐루 함수 적용하기 

합성곱 계산을 수행한다음 렐루 하뭇를 저굥하여 합성곱 층 완성

```python
def forpass(self,w):
	
	#렐루 함수를 적용한다.
	r_out = tf.nn.relu(c_out)
```

1. 풀링 적용하고 완전 연결층 수정하기

max_pool2d() 함수를 사용하여 2*2 크기의 풀링을 적용

이 단계에서 만들어지는 특성 맵의 크기는 14*14*10

풀링으로 특성 맵의 크기를 줄인 다음 tf.reshape() 함수를 사용해 일렬로 펼침. 이 때 배치 차원을 제외한 나머지 차원만 펼쳐야 함.

그 다음 코드는 완전 연결층에 해당.

conv2d() 와 max_pool2d() 등이 텐서 객체를 반환하기에 matmul()을 사용함.

```python
def forpass(self,x):
	
	#2X2 최대 풀링을 적용
	p_out = tf.nn.max_pool2d(r_out, ksize = 2, strides = 2, padding='VALID')
	#첫 번째 배치 차원을 제외하고 출력을 일렬로 펼침.
	f_out = tf.reshape(p_out, [x.shape[0], -1)
	z1 = tf.matmul(f_out, self.w1) + self.b1
	a1 = tf.nn.relu(z1)
	z2 = tf.matmul(a1, self.w2) + self.b2
	return z2
```

### 합성곱 신경망의 역방향 계산 구현하기

텐서플로의 자동 미분 기능 사용하기

- 자동 미분의 사용 방법 알아보기

```python
x = tf.Variable(np.array([1.,2.,3.]))
with tf.GradientTape() as tape :
	y=x**3+2*x+5

#그레디언트를 계산
print(tape.gradient(y,x))
```

자동 미분 기능을 사용하려면 with 블럭으로 tf.GradientTape() 객체가 감시할 코드를 감싸야 한다.

tape 객체는 with블럭 안에서 일어나는 모든 연산을 기록하고 텐서플로 변수인 tf.Variable 객체를 자동으로 추적한다.

그레디언트를 계산하려면 미분 대상 객체와 변수를 tape 객체의 gradient() 메서드에 전달해야한다. 

1. 역방향 계산 구현하기

MultiClassNetwork 클래스에서는 training() 메서드에서 backprop() 메서드를 호출하여 가중치를 업데이트 햇다.

하지만 자동 미분 기능을 사용하면 backprop() 메서드를 구현할 필요가 없다.

```python
def training(self, x, y):
	m = len(x) #샘플 개수를 저장
	with tf.GradientTape() as tape:
		z = self.forpass(x) # 정방향 계산을 수행
		#손실을 계산
		loss = tf.nn.softmax_cross_entropy_with_logits(y,z)
		loss = tf.reduce_mean(loss)
```

training() 메서드에서 forpass() 메서드를 호출해 정방향 계산 수행한 다음 tf.nn.softmax_cross_entropy_with_logits() 함수를 호출해 정방향 계산 결과와 타깃을 기반으로 손실값을 계산함.

이렇게 하면 크로스 엔트로피 손실과 그레디언트 계싼을 올바르게 처리해주므로 편리함.

이 때 softmax_cross_entropy_with_logits() 함수는 배치의 각 샘플에 대한 손실을 반환 ⇒ reduce_mean() 함수로 평균을 계산.

1. 그레디언트 계산하기

가중치와 절편을 업데이트 하기 - tape.gradient() 메서드를 사용해 그레디언트를 자동으로 계산할 수 있음.

conv_w, conv_b 를 포함해 그레디언트가 필요한 가중치를 리스트로 나열

optimizer.apply_gradients()메서드 등장

텐서플로가 여러 종류의 경사하강법 알고리즘을 클래스로 미리 구현해놓았기 때문에 경사 하강법 알고리즘을 바꾸어 가며 테스트 할 때 가중치를 업데이트 하는 코드를 일일이 고쳐야 한다면 번거롭다. 따라서 옵티마이저를 사용하면 간단하게 알고리즘 바꾸어 테스트할 수 있음.

apply_gradients() 메서드는 그레디언트와 가중치를 튜플로 묶은 리스트를 전달해야함.

파이썬의 zip 반복자를 사용해 이를 구현함.

```python
def training(self,x,y):
	weights_list = [self.conv_w, self.conv_b, self.w1, self.w2, self.b2]
	#가중치에 대한 그레디언트를 계산
	grads = tape.gradient(loss, weights_list)
	#가중치를 업데이트함.
	self.optimizer.apply_gradients(zip(grads,weights_list))
```

### 옵티마이저 객체를 만들어 가중치 초기화 하기

training() 메서드에 등장하는 self.optimizer를 fit() 메서드에서 만들어 본다.

1. fit() 메서드 수정하기

fit() 메서드는 옵티마이저 객체 생성 부분만 제외하면 MultiClassNetwork 클래스의 fit() 메서드와 거의 동일

텐서플로는 tf.optimizers 모듈 아래에 여러 종류의 경사하강법 구현해놨음.

SGD 옵티마이저 객체는 기본 경사하강법

아래는 SGD 옵티마이저 생성 코드 추가한 핏 메서드

```python
def fit(self, x,y, epochs=100, x_val = None, y_val = None):
	self.init_weights(x.shape, y.shape[1]) #은닉층과 출력층의 가중치 초기화
	self.optimizer = tf.optimizers.SGD(learning_rates = self.lr) 
	#epochs 만큼 반복
	for i in range(epochs):
		print('에포크', i, end='')
		#제너레이터 함수에서 반환한 미니 배치를 순환한다.
		batch_losses = []
		for x_batch, y_batch in self.gen_batch(x,y):
			print('.', end = '')
			self.training(x_batch, y_batch)
			#배치 손실을 기록
			batch_losses.append(self.get_loss(x_batch,y_batch))
			print()
			#배치 손실 평균 내어 훈련 손실값으로 저장
			self.losses.append(np.mean(batch_losses))
			#검증 세트에 대한 손실을 계산
			self.val_losses.append(slef.get_loss(x_val, y_val))
```

1. init_weights() 메서드 수정하기

가중치를 초기화하는 init_weights() 메서드는 큰 변화가 있음. 가중치를 glorot_uniform() 함수로 초기화한다는 점과 텐서플로의 자동 미분 기능을 사용하기 위해 가중치를 tf.Variable() 함수로 만들어야 한다는 점이다. 새로운 함수인 glorot_uniform()을 사용하는 이유는 : 

합성곱의 가중치와 완전 연결층의 가중치를 tf.Variable() 함수로 선언할 때 입력값에 따라 자료형이 자동으로 결정된다.

np.zeros() 함수는 기본적으로 64bit 실수를 만듦. 따라서 절편 변수를 가중치 변수와 동일하게 32 비트 실수로 맞추기 위해 dtype 매개변수에 float 지정함.

```python
def init_weights(self, input_shape, n_classes):
		g= tf.initializers.glorot_uniform()
		self.conv_w = tf.Variable(g(3,3,1,self.n_kernels)))
		self.conv_b = tf.Variable(np.zeros(self.n_kernels), dtype = float)
		n_features = 14*14*self.n_kernels
		self.w1 = tf.Variable(g((n_features, self.units))) #(특성 개수, 은닉층의 크기)
		self.b1 = tf.Variable(np.zeros(self.units), dtype = float) #은닉층의 크기
		self.w2 = tf.Variable(g((self.units, n_classes))) #(은닉층의 크기, 클래스 개수)
		self.b2 = tf.Variable(np.zeros(n_classes), dtype = float) # 클래스 개수
```

### glorot_uniform() 을 알아본다.

glorot_uniform() 함수는 가중치를 초기화할 때 글로럿 초기화라는 방법을 사용할 수 있게 해줌. 신경망 모델이 너무 커지면 손실함수가 복잡해져서 출발점에 따라 결과가 달라질 수 있음. 

글로럿 초기화 방식으로 가중치를 초기화 한다.

텐서플로의 glorot_uniform() 함수는 ~~ 사이에서 균등하게 난수를 발생시켜 가중치를 초기화한다.글로럿 초기화를 사용하는 방법은 간단하다. 이 함수에서 만든 객체를 호출할 때 필요한 가중치의 크기를 전달하면 됨.

위 코드를 참조해봐!

## 합성곱 신경망 훈련하기

1. 데이터 세트 불러오기

텐서플로 사용해 MNIST 데이터 세트 불러오기

```python
(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
```

1. 훈련 데이터 세트를 훈련 세트와 검증 세트로 나누기

사이킷런을 사용하여 훈련 데이터 세트를 훈련 세트와 검증 세트로 나눈다.

```python
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify =y_train_all, test_size=0.2, random_state = 42)
```

1. 타깃을 원-핫 인코딩으로 변환하기

y_train, y_val 은 정수로 이루어진 1차원 배열.

합성곱 신경망의 타깃으로 사용하려면 두 배열의 요소들을 원-핫 인코딩으로 변경해야함.

```python
y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val)
```

1. 입력 데이터 준비하기

합성곱 신경망은 입력 데이터를 일렬로 펼칠 필요가 없다. 높이와 너비 차원을 그대로 유지한 채 신경망에 주입.

마지막에 컬러 채널 추가해야함. ⇒ 넘파이에 reshape() 메서드를 사용하면 마지막 차원 간단하게 추가 가능

```python
x_train = x_train.reshape(-1,28,28,1)
x_val = x_val.reshape(-1,28,28,1)
```

1. 입력 데이터 표준화 전처리하기

입력 데이터가 이미지이기에 0~255 사이의 정수로 픽셀 강도를 표현함.

```python
x_train = x_train/255
x_val = x_val/255
```

1. 모델 훈련하기

훈련 준비를 마쳤기에, ConvolutionNetwork 클래스 객체를 생성한 다음 fit() 메서드를 호출해 모델을 훈련

합성곱 커널은 10개를 사용하고 완전 연결층의 뉴런은 100개를 사용한다.

배치 크기는 128개로 지정, 학습률은 0.01로 지정

```python
cn = ConvolutionNetwork(n_kernels = 10, units = 100, batch_size = 128, learning_rate = 0.01)
cn.fit(x_train, y_train_encoded, x_val = x_val, y_val = y_val_encoded, epochs = 20)
```

1. 훈련, 검증 손실 그래프 그리고 검증 세트의 정확도 학인하기

## 케라스로 합성곱 신경망 만들기

1. 필요한 클래스 임포트

tensorflow.keras.layers 에 포함된 Conv2d, Maxpooling2d, Flatten, Dense 클래스를 임포트 

1. 합성곱층 쌓기
2. 풀링층 쌓기
3. 완전 연결층에 주입할 수 있도록 특성 맵 펼치기
4. 완전 연결층 쌓기
5. 모델 구조 살펴보기

### 합성곱 신경망 모델 훈련하기

1. 모델 컴파일 후 크로스 엔트로피 손실함수 사용

```python
con1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
```

1. 아담 옵티마이저 사용하기
2. 손실 그래프와 정확도 그래프 확인하기

### 드롭아웃을 알아보기

; 과대적합을 줄이는 방법

텐서플로에서는 드롭아웃의 비율만큼 뉴런의 출력을 높임.

### 드롭아웃을 적용해 합성곱 신경망 구현

1. 케라스로 만든 합성곱 신경망에 드롭아웃 적용하기

합성곱층과 완전 연결층 사이에 드롭아웃층을 추가하여 과대적합이 어떻게 바뀌는지

1. 드롭아웃층 확인하기

summary() 메서드 사용

1. 훈련하기
2. 손실 그래프와 정확도 그래프 그리기
3.
