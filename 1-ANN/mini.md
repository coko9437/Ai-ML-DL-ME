
---

### **코드 종합 설명**

이 코드는 Scikit-learn의 **와인 품질 데이터셋**을 사용하여, 와인의 13가지 화학적 특성(입력)을 기반으로 와인의 품질 등급(출력)을 예측하는 **다층 퍼셉트론(MLP) 회귀 모델**을 구축하는 전체 과정을 보여줍니다.

1.  **데이터 준비 및 전처리:** `scikit-learn`에서 데이터를 불러온 뒤, `StandardScaler`를 이용해 각 특성의 스케일을 맞추는 **표준화**를 진행합니다. 그 후, 데이터를 학습용과 평가용으로 분할하여 모델의 일반화 성능을 측정할 준비를 합니다.
2.  **PyTorch 데이터 파이프라인 구축:** 전처리된 데이터를 PyTorch 텐서로 변환하고, `TensorDataset`과 `DataLoader`를 이용해 **미니배치(mini-batch)** 단위로 모델에 효율적으로 공급할 수 있는 데이터 파이프라인을 구축합니다.
3.  **MLP 모델 정의 및 학습:** `nn.Module`을 상속받아 2개의 은닉층을 가진 `Regressor` 모델을 정의합니다. 손실 함수로는 `MSELoss`, 최적화 알고리즘으로는 `Adam`을 사용하여 정의된 훈련 루프(Training Loop)에 따라 모델을 학습시킵니다.
4.  **모델 평가 및 예측:** 학습이 완료된 모델을 `model.eval()` 모드로 전환하여, 학습에 사용되지 않은 평가 데이터(Test set)에 대한 성능을 **평균 제곱 오차(MSE)**로 평가합니다. 또한, 완전히 새로운 가상의 샘플 데이터를 입력받아 품질을 예측하고, 결과를 해석하기 쉬운 1~9 등급의 정수로 변환하는 실용적인 예측 함수를 구현합니다.

이 미니 프로젝트는 실제 데이터 분석 문제에 딥러닝을 적용하는 표준적인 워크플로우(데이터 로딩 → 전처리 → 모델링 → 학습 → 평가 → 예측)를 압축적으로 보여주는 훌륭한 예제입니다.

---

### **Part 1: 데이터 준비 및 PyTorch 파이프라인 구축**

원본 데이터를 불러와 모델 학습에 적합한 형태로 정제하고, PyTorch의 `DataLoader`를 이용해 효율적인 학습 환경을 구축하는 단계입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

# 📌 1. 데이터 불러오기 및 기본 변환
wine = load_wine()
X = wine.data  # 입력 특성 (13개), Numpy 배열
y = wine.target.reshape(-1, 1) # 정답 (품종 0, 1, 2), Numpy 배열
# 이 예제에서는 품종을 품질 점수로 가정하여 회귀 문제로 변환합니다.
y = y * 3 + 1  # 0,1,2 -> 1,4,7 (실제로는 1~9 사이의 연속값으로 예측하는 회귀 문제로 설정)

# 📌 2. 데이터 전처리
# StandardScaler: 각 특성의 평균을 0, 표준편차를 1로 변환하여 스케일을 맞춥니다.
scaler = StandardScaler()
# fit_transform은 학습 데이터에 맞춰 스케일러를 '학습'시키고, 그 기준으로 데이터를 '변환'합니다.
X_scaled = scaler.fit_transform(X)

# train_test_split: 데이터를 학습용과 평가용으로 8:2 비율로 무작위 분할합니다.
# random_state=42는 재현성을 위해 난수 시드를 고정합니다.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 📌 3. PyTorch 텐서 변환 및 DataLoader 생성
# Numpy 배열을 PyTorch 텐서로 변환합니다.
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
# ... (X_test, y_test도 동일하게 변환) ...

# TensorDataset: 입력 텐서와 정답 텐서를 하나의 데이터셋으로 묶어줍니다.
train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)

# DataLoader: Dataset으로부터 데이터를 배치 크기만큼씩 꺼내주는 이터레이터(iterator)를 생성합니다.
# - batch_size=32: 한 번의 반복(iteration)에 32개의 데이터를 사용
# - shuffle=True: 매 에포크(epoch)마다 데이터 순서를 섞음
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
```

#### **2. 해당 설명**

이 파트는 실제 머신러닝 프로젝트의 **가장 많은 시간을 차지하는 데이터 준비 과정**을 보여줍니다. `StandardScaler`를 사용하는 **스케일링**은 매우 중요한 단계입니다. 만약 알코올 도수(0~20)와 말산(0~5)처럼 특성별로 값의 범위가 크게 다르면, 모델이 범위가 큰 특성에 과도하게 영향을 받아 학습이 불안정해질 수 있습니다. 스케일링은 모든 특성이 동등한 중요도로 출발하게 만들어 모델의 학습 효율과 성능을 높여줍니다.

**`DataLoader`**는 이전 예제의 `Custom Dataset`을 한 단계 더 발전시킨 PyTorch의 표준 데이터 공급 방식입니다. `DataLoader`를 `for`문에 사용하면, 우리는 복잡한 인덱싱 없이 매 반복마다 자동으로 섞이고 준비된 미니배치 데이터를 편리하게 얻을 수 있습니다. 이는 코드를 깔끔하게 유지하고 대용량 데이터셋을 효율적으로 처리하는 데 필수적입니다.

#### **3. 응용 가능한 예제**

**"이미지 데이터셋을 위한 DataLoader"**

`torchvision.datasets`에서 불러온 이미지 데이터셋(예: CIFAR-10)을 `DataLoader`에 전달하면, 이미지와 레이블로 구성된 미니배치를 자동으로 생성해줍니다. `num_workers` 옵션을 추가하면 여러 개의 CPU 코어를 사용해 데이터 로딩을 병렬 처리하여 학습 속도를 더욱 높일 수 있습니다.

#### **4. 추가하고 싶은 내용 (Validation Set의 부재)**

이 예제는 데이터를 학습(train)과 평가(test) 두 가지로만 나누었습니다. 하지만 더 엄밀한 모델 개발 과정에서는 학습(train), **검증(validation)**, 평가(test) 세 가지로 나눕니다. 학습 중에는 `validation set`으로 모델의 성능을 주기적으로 체크하여 최적의 에포크(epoch)를 찾거나 하이퍼파라미터(학습률 등)를 튜닝하는 데 사용하고, `test set`은 모든 학습과 튜닝이 끝난 뒤 **오직 단 한 번**, 최종 성능을 측정하는 데 사용합니다.

#### **5. 심화 내용 (`fit` vs `transform` vs `fit_transform`)**

`StandardScaler`와 같은 `scikit-learn`의 전처리기(Preprocessor)에는 세 가지 주요 메소드가 있습니다.
*   **`fit(data)`**: 데이터의 분포(평균, 표준편차 등)를 **학습**합니다.
*   **`transform(data)`**: 학습된 분포를 기준으로 데이터를 **변환**합니다.
*   **`fit_transform(data)`**: `fit`과 `transform`을 **동시에 수행**합니다.

**매우 중요한 규칙**은, `fit` 또는 `fit_transform`은 **반드시 학습 데이터(train set)에만 사용**해야 한다는 것입니다. 평가 데이터(test set)나 새로운 샘플 데이터에는 **학습 데이터로 `fit`된 스케일러를 그대로 가져와 `transform`만 적용**해야 합니다. 이는 평가 데이터의 정보(평균, 표준편차 등)가 학습 과정에 유출(leak)되는 것을 막고, 모든 데이터를 동일한 기준으로 변환하기 위함입니다.

---

### **Part 2: 모델 학습, 평가 및 실제 예측**

MLP 모델을 정의하고, 배치 단위로 학습을 진행한 뒤, 학습된 모델의 성능을 평가하고 새로운 데이터에 대한 예측을 수행하는 단계입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 📌 4. MLP 모델 정의
# 이전 예제와 구조는 유사하지만, 활성화 함수를 nn.Module로 정의하여 재사용성을 높였습니다.
class Regressor(nn.Module):
    # ... (__init__ 부분 생략) ...
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout은 활성화 함수 다음에 적용하는 것이 일반적입니다.
        x = self.relu(self.fc2(x))
        x = self.fc3(x)      # 회귀 문제의 출력층은 보통 선형 출력을 그대로 사용합니다.
        return x

# 📌 5~8. 모델 학습
model = Regressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
train_losses = []
# DataLoader를 for문에 직접 사용하여 배치 단위 학습을 진행합니다.
for epoch in range(num_epochs):
    epoch_loss = 0.0
    # train_loader는 매 반복마다 32개의 (X, y) 데이터 묶음을 반환합니다.
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X) # 32개의 입력에 대한 예측 수행
        loss = criterion(predictions, batch_y) # 32개의 예측과 정답으로 손실 계산
        loss.backward()
        optimizer.step()
        # loss.item()은 텐서에서 스칼라 값을 추출합니다. 배치 손실을 누적합니다.
        epoch_loss += loss.item()

    # 한 에포크가 끝나면, 누적된 배치 손실을 배치의 개수로 나누어 평균 에포크 손실을 계산합니다.
    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

# ... (손실 그래프 시각화 코드 생략) ...

# 📌 10. 모델 평가 및 예측
model.eval() # 평가 모드로 전환 (Dropout 비활성화)
with torch.no_grad(): # 기울기 계산 중지
    # 테스트 데이터 전체에 대한 예측을 한 번에 수행합니다.
    y_pred = model(X_test).numpy()

# sklearn.metrics의 mean_squared_error를 이용해 성능(MSE) 계산
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {test_mse:.4f}")

# ... (예측값 vs 실제값 시각화 코드 생략) ...

# ⭐ 샘플 데이터에 대한 실용적인 예측 함수
def predict_sample(sample):
    # 1. 샘플 데이터를 numpy 배열로 변환
    sample = np.array(sample).reshape(1, -1)
    # 2. '학습 데이터'로 학습된 scaler를 이용해 동일한 기준으로 변환
    sample_scaled = scaler.transform(sample)
    # 3. PyTorch 텐서로 변환
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        # 4. 모델로 예측 수행
        predicted_quality = model(sample_tensor).item()

    # 5. 후처리: 예측 결과를 해석하기 쉬운 정수로 변환하고, 1~9 범위를 벗어나지 않도록 클리핑(clipping)
    predicted_quality_int = min(9, max(1, round(predicted_quality)))
    return predicted_quality, predicted_quality_int

# ... (샘플 예측 실행 코드 생략) ...
```

#### **2. 해당 설명**

이 파트는 학습된 모델의 **실용적인 활용**에 초점을 맞춥니다. 학습 루프는 `DataLoader`를 사용하여 더 표준적인 배치 단위 학습 형태로 구현되었습니다. 한 에포크는 `DataLoader`가 가진 모든 미니배치를 한 번씩 다 사용하는 것을 의미합니다.

**모델 평가** 부분에서는 `model.eval()`과 `torch.no_grad()`를 사용하는 것이 핵심입니다. 이는 모델을 '시험 모드'로 전환하여, 드롭아웃 같은 훈련용 기술을 끄고 불필요한 기울기 계산을 생략하여 정확하고 효율적인 평가를 가능하게 합니다.

가장 중요한 부분은 **`predict_sample` 함수**입니다. 이 함수는 새로운 데이터가 들어왔을 때 수행해야 할 일련의 과정을 명확히 보여줍니다.
1.  **동일한 전처리 적용:** 새로운 데이터도 **반드시 학습 데이터를 기준으로 학습된 `scaler`를 이용해 변환**해야 합니다.
2.  **예측:** `model.eval()` 상태에서 예측을 수행합니다.
3.  **후처리(Post-processing):** 모델이 내놓은 실수(float) 형태의 예측값은 그대로 사용하기 어려울 수 있습니다. 이 값을 `round()`를 이용해 반올림하고 `min`, `max`를 이용해 1~9 사이의 값으로 보정해주는 **후처리** 과정은, 모델의 예측을 실제 비즈니스 문제에 적용 가능한 형태로 만드는 중요한 단계입니다.

#### **3. 응용 가능한 예제**

**"스팸 메일 분류기 서빙(Serving)"**

사용자로부터 새로운 이메일 텍스트가 입력되면, ① 학습 때 사용했던 것과 동일한 단어 토큰화 및 벡터화 전처리 과정을 거친 뒤, ② `model.eval()` 상태의 학습된 분류 모델에 입력하여 스팸 확률을 예측하고, ③ 예측된 확률이 특정 임계값(예: 0.8)을 넘으면 '스팸', 아니면 '정상'으로 최종 판정하여 사용자에게 결과를 보여주는 서비스를 만들 수 있습니다.

#### **4. 추가하고 싶은 내용 (모델의 불확실성)**

예측 함수에서 `predicted_quality_int = min(9, max(1, round(predicted_quality)))` 와 같이 결과를 정수로 변환하는 것은 사용자의 편의를 위한 것입니다. 하지만 `predicted_quality`라는 실수 값 자체도 중요한 정보를 담고 있습니다. 예를 들어, 예측값이 4.51이라면 모델은 4등급과 5등급 사이에서 꽤 헷갈리고 있다는 의미이며, 4.95라면 5등급에 매우 가깝다고 확신하고 있다는 의미입니다. 이처럼 모델의 출력값(logit) 자체는 모델의 **확신도(confidence)**를 나타내는 지표로 활용될 수 있습니다.

#### **5. 심화 내용 (회귀 모델 평가 지표)**

MSE는 회귀 모델의 성능을 평가하는 가장 일반적인 지표지만, 단점이 있습니다. 오차를 '제곱'하기 때문에, 다른 예측은 다 잘 맞췄는데 유독 하나의 예측에서 큰 오차를 낸 경우 페널티가 매우 커집니다.
*   **RMSE (Root Mean Squared Error):** MSE에 루트를 씌운 값입니다. 단위가 원래 데이터와 동일해져서 해석이 더 직관적입니다. (예: 주택 가격 오차가 10,000 (달러^2) -> RMSE는 100 (달러))
*   **MAE (Mean Absolute Error):** 오차의 '절대값'의 평균을 사용합니다. RMSE와 달리 이상치(outlier)에 덜 민감하여, 예측 오차의 일반적인 크기를 파악하는 데 유용합니다.

어떤 평가 지표를 사용할지는 문제의 특성과 무엇을 중요하게 볼 것인지에 따라 달라집니다.