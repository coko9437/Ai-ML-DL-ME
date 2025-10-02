
---

### **코드 종합 설명**

이 코드는 **인공 신경망(Artificial Neural Network)**의 가장 기본적인 형태를 PyTorch의 `nn.Module`을 사용하여 직접 구축하고 학습시키는 과정을 보여줍니다. 전체적인 흐름은 다음과 같습니다.

1.  **단층 퍼셉트론 (Single-Layer Perceptron):** `nn.Linear` 하나만으로 구성된 가장 단순한 신경망(선형 회귀 모델)을 만들어, 신경망 모델의 기본 구조(`__init__`, `forward`)와 훈련 루프(training loop)의 작동 방식을 이해합니다. 이는 사실상 하나의 뉴런(시냅스)이 어떻게 학습하는지를 보여주는 예제입니다.
2.  **다층 퍼셉트론 (Multi-Layer Perceptron, MLP):** 주택 가격 예측과 같은 더 복잡한 실제 데이터를 다루기 위해, 여러 개의 선형 계층(`nn.Linear`)과 비선형 활성화 함수(`F.relu`)를 쌓아 만든 다층 신경망 모델을 구축합니다.
3.  **데이터 처리 및 모델링 파이프라인:** 실제 CSV 파일로부터 데이터를 불러와 `pandas`로 탐색하고, `scikit-learn`으로 학습/평가 데이터로 분할하며, PyTorch의 `Dataset`과 `DataLoader`를 사용해 모델에 효율적으로 공급하는 표준적인 머신러닝/딥러NING 파이프라인을 학습합니다.
4.  **모델 성능 향상 기법:** 학습 과정에서 **과적합(Overfitting)**을 방지하기 위한 대표적인 규제(Regularization) 기법인 **드롭아웃(Dropout)**의 개념과 사용법을 익힙니다. 또한, 학습이 끝난 모델을 올바르게 **평가(Evaluation)**하는 방법과 그 중요성을 배웁니다.

이 코드를 통해, 단순한 수학적 모델에서 시작하여 여러 개의 '뉴런' 층을 쌓아 복잡한 데이터의 패턴을 학습하는 '인공 두뇌'의 기본 원리를 이해할 수 있습니다.

---

### **Part 1: 단층 퍼셉트론 - 가장 단순한 신경망 (선형 회귀)**

하나의 뉴런(또는 하나의 선형 계층)으로 구성된 가장 기본적인 신경망을 통해 모델을 클래스로 정의하는 방법과 훈련 과정을 이해합니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# 📌 1. 데이터 준비
# y = 2x + noise 형태의 선형 관계를 갖는 가상 데이터 생성
x = torch.FloatTensor(range(5)).unsqueeze(1) # 입력, Shape: [5, 1]
y = 2 * x + torch.rand(5, 1)                  # 정답, Shape: [5, 1]

# 📌 2. 모델 클래스 정의
# 모든 PyTorch 모델은 nn.Module 클래스를 상속받아 만들어야 합니다.
class LinearRegressor(nn.Module):
    # __init__ (생성자): 모델에 필요한 구성 요소(계층, layer)들을 정의하는 곳
    def __init__(self):
        # super().__init__()는 nn.Module의 생성자를 먼저 호출하는 필수 코드입니다.
        super().__init__()
        # self.fc는 'Fully Connected'의 약자로, nn.Linear 계층의 인스턴스를 생성하여 저장합니다.
        # nn.Linear(in_features, out_features, bias=True)
        # - in_features=1: 입력 데이터의 특성(feature) 개수
        # - out_features=1: 출력 데이터의 특성 개수
        # - bias=True: 편향(b)을 학습할지 여부 (기본값)
        self.fc = nn.Linear(1, 1, bias=True)

    # forward: 데이터가 모델에 입력되었을 때, 어떤 계산을 거쳐 출력을 만들지 데이터의 흐름을 정의하는 곳
    def forward(self, x):
        # 입력 x를 self.fc 계층에 통과시켜 예측값 y를 계산합니다.
        # 내부적으로 y = Wx + b 연산이 수행됩니다.
        y_pred = self.fc(x)
        return y_pred

# 📌 3. 모델, 손실 함수, 옵티마이저 초기화
model = LinearRegressor() # 위에서 정의한 클래스의 인스턴스를 생성하여 모델 객체로 사용
learning_rate = 1e-3
criterion = nn.MSELoss() # 손실 함수 (채점 기준)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # 최적화 알고리즘 (오답노트 및 학습 방법)

# 📌 4. 훈련 루프
loss_stack = []
for epoch in range(1001):
    optimizer.zero_grad()   # 1. 기울기 초기화
    y_hat = model(x)        # 2. 예측 (model.forward(x)가 호출됨)
    loss = criterion(y_hat, y) # 3. 손실 계산
    loss.backward()         # 4. 역전파
    optimizer.step()        # 5. 파라미터 업데이트
    loss_stack.append(loss.item())
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}:{loss.item():.4f}')
# ... (시각화 코드 생략) ...
```

#### **2. 해당 설명**

이전 예제에서는 `w`와 `b`를 직접 텐서로 선언했지만, 이 코드에서는 `nn.Module`과 `nn.Linear`를 사용하여 훨씬 더 체계적이고 확장 가능한 방식으로 모델을 정의합니다. 이것이 바로 PyTorch의 표준 모델 구축 방식입니다.

*   **`__init__`**: 모델이라는 '기계'를 만들기 위해 필요한 '부품'(레이어)들을 미리 준비해두는 공간입니다.
*   **`forward`**: 데이터라는 '재료'가 들어왔을 때, 준비된 '부품'들을 어떤 순서로 조립하여 '제품'(예측값)을 만들지 그 '설계도'를 그리는 공간입니다.

이러한 클래스 기반의 모델링은 나중에 복잡한 다층 신경망을 만들 때 코드의 재사용성과 가독성을 크게 높여줍니다. 훈련 루프의 5단계(초기화-예측-손실-역전파-업데이트)는 이전 예제와 동일하며, 모든 지도 학습 모델의 기본 골격이 됩니다.

#### **3. 응용 가능한 예제**

**"키에 따른 몸무게 예측 모델"**

`X` 데이터를 '키'로, `y` 데이터를 '몸무게'로 설정하고 `LinearRegressor` 클래스를 그대로 사용하면, 키와 몸무게 사이의 선형 관계를 학습하는 모델을 만들 수 있습니다.

#### **4. 추가하고 싶은 내용 (모델 파라미터 확인)**

`model.parameters()`는 옵티마이저에게 학습할 대상을 알려주는 역할뿐만 아니라, 현재 모델이 어떤 파라미터를 가지고 있는지 직접 확인할 수도 있습니다.

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data, param.grad)
```
이 코드를 실행하면 `fc.weight`와 `fc.bias`의 현재 값과, `loss.backward()` 이후 계산된 기울기 값을 볼 수 있습니다.

#### **5. 심화 내용 (객체 지향 프로그래밍과 `nn.Module`)**

PyTorch의 모델링 방식은 **객체 지향 프로그래밍(OOP)**에 기반을 두고 있습니다. `nn.Module`이라는 '설계도(클래스)'를 상속받아 우리만의 새로운 '기계(모델)'를 만드는 것입니다. 이 방식의 가장 큰 장점은 **모듈화**입니다. 복잡한 모델을 만들 때, 작은 기능 단위(예: 합성곱 블록, 어텐션 모듈)를 별도의 `nn.Module` 클래스로 만들어두고, 더 큰 모델에서 이 작은 모듈들을 부품처럼 가져와 조립할 수 있습니다. 이는 코드의 유지보수와 재사용을 매우 용이하게 합니다.

---

### **Part 2: 다층 퍼셉트론(MLP) - 실제 데이터로 만드는 복잡한 신경망**

여러 개의 뉴런 층(은닉층)을 쌓아, 단순한 선형 관계를 넘어선 복잡한 패턴을 학습하는 다층 퍼셉트론을 구현합니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
# 📌 1. 데이터 준비: Custom Dataset 및 DataLoader 생성
# ... (pandas, numpy, train_test_split import 생략) ...
# csv 파일에서 데이터를 불러와 X(입력 특성), Y(정답)로 분리

# PyTorch의 Dataset 클래스를 상속받아 우리만의 데이터셋 형식을 정의합니다.
class TensorData(Dataset):
    # 데이터셋이 처음 생성될 때 한 번 실행됩니다.
    def __init__(self, x_data, y_data):
        # 입력과 정답 데이터를 float 타입의 텐서로 변환하여 저장합니다.
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        # 데이터의 총 길이를 저장해둡니다.
        self.len = self.y_data.shape[0]

    # 데이터셋에서 특정 인덱스(index)의 데이터를 요청받았을 때 실행됩니다.
    # DataLoader가 이 함수를 호출하여 데이터를 가져옵니다.
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # 데이터셋의 총 길이를 반환합니다.
    def __len__(self):
        return self.len

# 학습/평가용으로 나눈 numpy 배열을 이용해 TensorData 인스턴스를 생성
trainsets = TensorData(X_train, Y_train)
# TensorData를 DataLoader에 전달하여 배치 단위로 데이터를 공급할 준비
trainloader = torch.utils.data.DataLoader(trainsets, batch_size=32, shuffle=True)
# ... (testloader도 동일하게 생성) ...

# 📌 2. 모델 클래스 정의: 다층 퍼셉트론 (MLP)
import torch.nn.functional as F # relu와 같은 함수 형태의 기능들을 포함

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        # 3개의 선형 계층을 정의
        self.fc1 = nn.Linear(13, 50) # 입력 특성 13개 -> 첫 번째 은닉층 뉴런 50개
        self.fc2 = nn.Linear(50, 30) # 첫 은닉층 50개 -> 두 번째 은닉층 뉴런 30개
        self.fc3 = nn.Linear(30, 1)  # 두 번째 은닉층 30개 -> 출력 1개 (예측 가격)
        # Dropout 계층 정의: 훈련 중에 50%의 뉴런을 랜덤하게 비활성화시켜 과적합을 방지
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 데이터의 흐름을 정의
        # 1. fc1 통과 후, 활성화 함수 ReLU 적용
        x = F.relu(self.fc1(x))
        # 2. fc2 통과 후, ReLU 적용, 그 다음 Dropout 적용
        # dropout은 훈련(model.train() 모드) 중에만 작동하고, 평가(model.eval() 모드) 중에는 자동으로 비활성화됩니다.
        x = self.dropout(F.relu(self.fc2(x)))
        # 3. fc3 통과 (회귀 문제의 마지막 출력층에는 보통 활성화 함수를 적용하지 않음)
        x = self.fc3(x)
        return x

# 📌 3. 모델 평가 함수 정의
def evaluation(dataloader):
    predictions = torch.tensor([], dtype=torch.float)
    actual = torch.tensor([], dtype=torch.float)

    with torch.no_grad(): # 기울기 계산 비활성화
        model.eval() # 모델을 '평가 모드'로 전환. Dropout, BatchNorm 등의 동작을 변경.
        for data in dataloader:
            inputs, values = data
            outputs = model(inputs)
            # cat을 이용해 배치 단위의 예측값과 실제값을 하나의 큰 텐서로 누적
            predictions = torch.cat((predictions, outputs), 0)
            actual = torch.cat((actual, values), 0)
    
    # ... (RMSE 계산 부분 생략) ...
    return rmse
```

#### **2. 해당 설명**

이 파트는 **은닉층(Hidden Layer)**과 **드롭아웃(Dropout)**이라는 두 가지 중요한 개념을 도입합니다.
*   **은닉층:** 텍스트의 비유처럼, 은닉층을 추가하는 것은 모델이 데이터로부터 더 복잡하고 추상적인 특징을 학습할 수 있게 해주는 것과 같습니다. `fc1`은 원본 데이터에서 1차적인 특징을, `fc2`는 `fc1`이 만든 특징들을 조합하여 더 고차원적인 특징을 학습합니다. 선형 계층 사이에 **비선형 활성화 함수(`F.relu`)**를 추가해야만, 여러 층을 쌓는 것이 의미가 있습니다. 이것이 바로 모델의 표현력을 높여 'Deep'한 학습을 가능하게 하는 핵심 원리입니다.
*   **드롭아웃:** 훈련 과정에서 일부러 뉴런의 일부를 무작위로 '쉬게' 만드는 규제 기법입니다. 이는 모델이 특정 뉴런에 과도하게 의존하는 것을 방지하고, 여러 뉴런이 협력하여 더 강건한 특징을 학습하도록 유도합니다. 마치 팀 프로젝트에서 몇몇 에이스 멤버에게만 의존하지 않고 모든 팀원이 제 역할을 하도록 훈련시키는 것과 같습니다. 이를 통해 모델이 훈련 데이터는 잘 맞추지만 새로운 데이터는 못 맞추는 **과적합(Overfitting)**을 완화할 수 있습니다.

#### **3. 응용 가능한 예제**

**"이미지 분류를 위한 다층 퍼셉트론"**

28x28 크기의 MNIST 손글씨 숫자 이미지가 있을 때, 이미지를 `view(-1, 28*28)`을 이용해 784개의 픽셀 값을 가진 1차원 벡터로 펼칩니다. 그 후, 입력 특성이 784개이고, 출력은 10개(0~9 숫자 클래스)인 다층 퍼셉트론 모델을 설계하여 이미지 분류를 수행할 수 있습니다.

#### **4. 추가하고 싶은 내용 (모델 `eval()` 모드의 중요성)**

`model.eval()`을 호출하는 것은 평가 시에 매우 중요합니다. 이 함수는 모델에게 "지금은 시험 볼 시간이야, 훈련 때 쓰던 꼼수(드롭아웃)는 쓰지 마!"라고 알려주는 스위치와 같습니다. `model.eval()` 모드에서는 드롭아웃이 비활성화되어 모든 뉴런이 예측에 참여하고, 배치 정규화(Batch Normalization)와 같은 다른 계층들도 훈련 때와는 다르게 동작합니다. 이를 설정하지 않으면 평가 결과가 부정확하게 나오므로, 훈련이 끝나고 평가를 시작하기 전에는 반드시 `model.eval()`을, 다시 훈련을 시작할 때는 `model.train()`을 호출해야 합니다.

#### **5. 심화 내용 (Adam 옵티마이저와 `weight_decay`)**

예제에서는 SGD보다 더 발전된 **Adam** 옵티마이저를 사용했습니다. Adam은 각 파라미터마다 개별적인 학습률을 적응적으로 조절해주고, 학습 과정의 '관성'(Momentum)을 고려하여 더 빠르고 안정적으로 최적점에 도달하는 경향이 있습니다.

`weight_decay`는 **L2 정규화(L2 Regularization)**를 구현하는 파라미터입니다. 이는 손실 함수에 '가중치들의 제곱 합'이라는 페널티 항을 추가하는 것과 같습니다. 모델이 학습할 때, 예측 오차뿐만 아니라 가중치 값 자체도 작게 유지하도록 강제합니다. 가중치 값이 너무 커지는 것을 막아 모델이 훈련 데이터에 너무 복잡하게 들어맞는(과적합) 현상을 방지하는 또 다른 강력한 규제 기법입니다.