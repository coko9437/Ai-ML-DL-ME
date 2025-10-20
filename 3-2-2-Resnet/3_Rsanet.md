---

### **코드 종합 설명**

제공해주신 코드는 구글 코랩(Google Colab) 환경에서 PyTorch를 사용하여 **전이 학습(Transfer Learning)**을 수행하는 전체 과정을 담고 있습니다. 이미지넷(ImageNet)이라는 대규모 데이터셋으로 **사전 학습된(pre-trained) ResNet-50 모델**을 불러온 후, 사용자가 가지고 있는 새로운 이미지 데이터셋을 분류하도록 재학습(Fine-tuning)시키는 것이 목표입니다.

코드의 주요 흐름은 다음과 같습니다.
1.  **환경 설정 및 데이터 준비:** 구글 드라이브를 연결하고, `ImageFolder`와 `DataLoader`를 이용해 이미지 데이터를 모델에 입력할 수 있는 형태로 불러오고 전처리합니다.
2.  **모델 구성:** PyTorch에서 제공하는 사전 학습된 ResNet-50 모델을 가져와서, 마지막 출력층(Fully Connected Layer)을 내 데이터셋의 클래스 개수에 맞게 수정합니다.
3.  **학습 및 저장:** 손실 함수(Loss Function)와 최적화 알고리즘(Optimizer)을 정의하고, 데이터를 반복적으로 모델에 통과시켜 가중치를 업데이트(학습)한 뒤, 학습된 모델을 파일로 저장합니다.
4.  **추론(예측):** 저장된 모델을 다시 불러와 새로운 샘플 이미지에 대해 어떤 클래스인지 예측하고, 그 결과를 시각화하여 보여줍니다.

함께 제공된 텍스트 파일은 깊은 신경망의 학습 어려움(기울기 소실)을 해결하기 위해 고안된 ResNet의 핵심 아이디어인 **'잔차 연결(Residual Connection)'**을 비유와 수식, 간단한 코드로 쉽게 설명하고 있습니다.

---

### **Part 1: 데이터 로드 및 전처리**

딥러닝 모델 학습의 첫 단계는 원본 이미지 데이터를 모델이 이해할 수 있는 숫자인 텐서(Tensor) 형태로 변환하고, 효율적인 학습을 위해 데이터를 묶음(Batch) 단위로 공급할 준비를 하는 것입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os

# ✅ 데이터 경로 설정
# 구글 드라이브 내의 이미지 데이터셋 폴더 경로입니다.
# 이 폴더 안에는 각 클래스 이름(예: 'cat', 'dog')으로 된 하위 폴더들이 있고,
# 그 안에 해당 클래스의 이미지가 들어있는 구조여야 합니다.
data_dir = "/content/drive/MyDrive/busanit501-1234/class"

# 폴더 안에 있는 클래스 확인 (디버깅용)
print("🔍 데이터셋 폴더 내 클래스 확인:")
# os.listdir은 해당 경로에 있는 파일 및 폴더 목록을 리스트로 반환합니다.
print(os.listdir(data_dir))

# 데이터 변환 (전처리) 파이프라인 정의
# transforms.Compose는 여러 변환 작업을 순서대로 묶어줍니다.
transform = transforms.Compose([
    # 1. 이미지 크기 조정: 모든 이미지를 (224, 224) 크기로 변경합니다.
    # ResNet 등 대부분의 사전 학습된 모델은 이 크기의 입력을 기대합니다.
    transforms.Resize((224, 224)),

    # 2. 텐서 변환: PIL 이미지(0~255 범위)를 PyTorch 텐서(0~1 범위, (C, H, W) 형태)로 변환합니다.
    transforms.ToTensor(),

    # 3. 정규화(Normalization): 각 채널(R, G, B)별로 평균을 빼고 표준편차로 나눕니다.
    # 아래 값들은 ImageNet 데이터셋의 통계값으로, ResNet 사용 시 표준적으로 사용됩니다.
    # 이를 통해 모델 학습이 더 빠르고 안정적으로 수렴하게 됩니다.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 객체 생성
# ImageFolder는 폴더 구조를 기반으로 이미지와 레이블(클래스 인덱스)을 자동으로 매핑해줍니다.
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 데이터 로더 생성
# DataLoader는 데이터셋에서 데이터를 배치 단위로 가져오고, 섞고(shuffle), 병렬 처리하는 역할을 합니다.
# batch_size=32: 한 번의 학습 단계(step)에 32개의 이미지를 사용합니다.
# shuffle=True: 매 에폭(epoch)마다 데이터 순서를 무작위로 섞어 학습 편향을 방지합니다.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 클래스 목록 및 개수 확인
# ImageFolder는 폴더 이름을 알파벳 순으로 정렬하여 클래스 리스트를 만듭니다.
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"✅ 클래스 목록: {class_names}, 총 {num_classes}개")
```

#### **2. 해당 설명**

**전처리(Preprocessing)**, 특히 **정규화(Normalization)**는 매우 중요합니다. 입력 데이터 값의 범위가 제각각이면 모델의 가중치가 불안정하게 학습될 수 있습니다. ImageNet 데이터셋의 평균과 표준편차를 사용하여 입력 데이터를 정규화하면, 데이터 분포가 사전 학습된 모델이 학습했던 분포와 비슷해져 전이 학습의 성능이 향상됩니다.

**`ImageFolder`와 `DataLoader`** 조합은 PyTorch에서 이미지 데이터를 다루는 표준적이고 강력한 패턴입니다. 복잡한 파일 입출력 코드를 직접 짤 필요 없이 폴더 구조만으로 라벨링된 데이터셋을 쉽게 준비할 수 있게 해줍니다.

#### **3. 응용 가능한 예제**

**"데이터 증강(Data Augmentation) 적용"**

학습 데이터가 부족할 때, 이미지를 무작위로 회전하거나 뒤집어서 데이터 양을 늘리는 기법입니다. `transform` 정의 부분에 추가하면 됩니다.
```python
transform = transforms.Compose([
    transforms.Resize((256, 256)), # 먼저 넉넉하게 크기 조절
    transforms.RandomCrop(224),    # 무작위 위치를 224 크기로 자름
    transforms.RandomHorizontalFlip(), # 50% 확률로 좌우 반전
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```
이렇게 하면 매 에폭마다 조금씩 다른 이미지가 모델에 입력되어 과적합(Overfitting)을 방지하고 모델의 일반화 성능을 높일 수 있습니다.

#### **4. 추가하고 싶은 내용 (`num_workers`)**

`DataLoader` 생성 시 `num_workers` 옵션을 사용하면 데이터 로딩에 사용할 CPU 프로세스(또는 쓰레드) 개수를 지정할 수 있습니다. GPU는 계산이 빠른데 데이터 로딩이 느려 병목이 생길 때, `num_workers=4` 처럼 설정하여 데이터 로딩 속도를 높일 수 있습니다. (단, Windows 환경에서는 오류가 발생할 수 있어 주의가 필요합니다.)

---

### **Part 2: ResNet 모델 구성 및 수정 (전이 학습)**

미리 학습된 고성능 모델을 가져와서, 내 문제(새로운 데이터셋 분류)를 해결할 수 있도록 모델의 구조를 일부 변경하는 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# ✅ ResNet-50 모델 불러오기
# pretrained=True (최신 버전에서는 weights='ResNet50_Weights.DEFAULT' 권장) 옵션은
# ImageNet 데이터(1000개 클래스)로 미리 학습된 가중치를 함께 다운로드합니다.
# 이 가중치들은 이미지의 선, 질감, 패턴 등 일반적인 특징을 추출하는 능력을 이미 가지고 있습니다.
model = models.resnet50(pretrained=True)

# ✅ 마지막 Fully Connected Layer (분류기) 변경
# 원본 ResNet-50의 마지막 층(model.fc)은 1000개의 클래스에 대한 확률을 출력합니다.
# 이를 내 데이터셋의 클래스 개수(num_classes)만큼 출력하도록 새로운 선형 층으로 교체합니다.
# model.fc.in_features: 직전 층에서 들어오는 입력 특징의 개수(ResNet50은 2048)를 그대로 유지합니다.
# 새로 교체된 층의 가중치는 랜덤하게 초기화되며, 이 부분 위주로 학습이 진행됩니다.
model.fc = nn.Linear(model.fc.in_features, num_classes)

# ✅ 모델을 연산 장치(GPU 또는 CPU)로 이동
# torch.cuda.is_available(): GPU 사용 가능 여부를 확인합니다.
# 모델(.to(device))과 데이터가 같은 장치에 있어야 연산이 가능합니다.
# 딥러닝 학습은 GPU에서 훨씬 빠르므로 필수에 가깝습니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ 모델이 {device}로 이동되었습니다.")

# 손실 함수 (Loss Function) 정의
# 다중 클래스 분류 문제이므로 CrossEntropyLoss를 사용합니다.
# 모델의 출력(예측)과 실제 정답(레이블) 사이의 차이를 계산합니다.
criterion = nn.CrossEntropyLoss()

# 최적화 함수 (Optimizer) 정의
# Adam 알고리즘을 사용하여 손실 함수를 최소화하는 방향으로 모델의 가중치(model.parameters())를 업데이트합니다.
# lr=0.001: 학습률(Learning Rate). 한 번의 업데이트 때 가중치를 얼마나 변경할지 결정합니다.
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

#### **2. 해당 설명**

이것이 바로 **전이 학습(Transfer Learning)**의 핵심입니다. 수백만 장의 이미지로 며칠 동안 학습해야 얻을 수 있는 '특징 추출 능력'을 공짜로 가져다 쓰는 것입니다. 특징 추출 부분(Convolutional layers)은 그대로 두고, 마지막 분류기(`model.fc`)만 내 데이터에 맞게 바꿔서 학습시키면 적은 데이터와 짧은 시간으로도 높은 성능을 낼 수 있습니다.

함께 제공된 텍스트 파일에서 설명한 **ResNet의 잔차 연결(Skip Connection)** 덕분에, 50개 층(`resnet50`)이라는 깊은 구조임에도 불구하고 기울기 소실 문제 없이 효과적으로 특징을 추출할 수 있습니다. $Y = F(X) + X$ 구조를 통해 그래디언트가 $X$ 경로를 타고 직접적으로 전달되기 때문입니다.

#### **3. 응용 가능한 예제**

**"특징 추출기 동결(Freezing)"**

데이터가 매우 적을 때는 앞부분의 특징 추출기 가중치까지 학습시키면 오히려 과적합이 발생할 수 있습니다. 이럴 때는 앞부분 가중치는 고정(동결)하고 마지막 분류기만 학습시킵니다.
```python
model = models.resnet50(pretrained=True)
# 모든 파라미터의 기울기 계산을 끔 (학습되지 않음)
for param in model.parameters():
    param.requires_grad = False

# 마지막 층을 새로 교체 (새로 만든 층은 기본적으로 requires_grad=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# 옵티마이저에는 학습할 파라미터(마지막 층)만 전달
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

#### **4. 심화 내용 (ResNet의 병목 구조 - Bottleneck)**

ResNet-50 이상부터는 연산량을 줄이기 위해 $1 \times 1$, $3 \times 3$, $1 \times 1$ 합성곱 층으로 구성된 **병목(Bottleneck) 블록**을 사용합니다. 첫 $1 \times 1$ 합성곱으로 채널 수를 줄였다가, $3 \times 3$ 합성곱 수행 후, 마지막 $1 \times 1$ 합성곱으로 다시 채널 수를 늘리는 방식입니다. 이를 통해 깊이는 유지하면서 파라미터 수와 연산량을 효율적으로 관리합니다.

---

### **Part 3: 모델 학습(Training) 및 저장**

준비된 데이터와 모델, 손실 함수, 옵티마이저를 사용하여 실제로 모델의 가중치를 업데이트하는 반복적인 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
num_epochs = 10  # 전체 데이터셋을 10번 반복해서 학습합니다.
print("🚀 모델 학습 시작...")

# 1. 에폭(Epoch) 루프: 전체 데이터셋에 대한 반복
for epoch in range(num_epochs):
    running_loss = 0.0 # 출력을 위해 손실 값을 누적할 변수

    # 2. 배치(Batch) 루프: 데이터 로더에서 미니배치 단위로 데이터를 가져옵니다.
    # enumerate를 사용하여 현재 배치의 인덱스(i)도 함께 받습니다.
    for i, (images, labels) in enumerate(train_loader):
        # 3. 데이터를 모델과 같은 장치(GPU)로 이동
        images, labels = images.to(device), labels.to(device)

        # 4. 기울기(Gradient) 초기화
        # 이전에 계산된 기울기 값이 남아있지 않도록 0으로 만듭니다.
        # PyTorch는 기울기를 누적하는 방식이므로 매 단계마다 필수입니다.
        optimizer.zero_grad()

        # 5. 순전파(Forward Pass): 모델에 이미지를 입력하여 예측값을 얻습니다.
        outputs = model(images)

        # 6. 손실(Loss) 계산: 예측값과 실제 정답을 비교하여 오차를 계산합니다.
        loss = criterion(outputs, labels)

        # 7. 역전파(Backward Pass): 손실에 대한 각 가중치의 기울기를 계산합니다 (자동 미분).
        loss.backward()

        # 8. 가중치 업데이트(Optimization step): 계산된 기울기와 학습률을 이용해 가중치를 수정합니다.
        optimizer.step()

        # 손실값 누적 (출력용, .item()으로 텐서에서 스칼라 값만 추출)
        running_loss += loss.item()

        # 로그 출력 코드 (현재 코드는 매 스텝마다 출력하도록 되어 있습니다.)
        # 보통은 일정 스텝마다 출력하거나 에폭 단위로 출력합니다.
        # 예시 코드의 의도를 살려 100 스텝마다 평균 손실을 출력하는 형태로 주석 설명
        # if (i + 1) % 100 == 0: # 100번째 배치마다
        #     print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
        #     running_loss = 0.0
        
    # (코드 원본대로 매 스텝 출력 시 로그가 너무 많아지므로, 에폭 종료 시 출력하는 것이 일반적입니다.)
    print(f"Epoch [{epoch+1}/{num_epochs}] 완료. 평균 Loss: {running_loss / len(train_loader):.4f}")

# %%
# 학습된 모델 저장
# 전체 모델을 저장하는 것보다 학습된 파라미터(state_dict)만 저장하는 것이 권장됩니다.
model_path = "/content/drive/MyDrive/busanit501-1234/resnet50_model.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ 학습된 모델 파라미터가 저장되었습니다: {model_path}")
```

#### **2. 해당 설명**

이 **`zero_grad()` -> `forward()` -> `loss calculation` -> `backward()` -> `step()`** 루프는 PyTorch 학습의 표준 공식입니다.
1.  이전 찌꺼기 청소하고 (`zero_grad`)
2.  문제 풀어서 답안 제출하고 (`forward`)
3.  채점해서 틀린 정도 확인하고 (`loss`)
4.  어디서 틀렸는지 오답 노트 만들고 (`backward`)
5.  공부 방법 수정하기 (`step`)
라고 이해하면 쉽습니다. ResNet의 스킵 연결 덕분에 `backward()` 과정에서 기울기가 소실되지 않고 입력층 가까이까지 잘 전달되어 깊은 네트워크도 효과적으로 학습됩니다.

#### **3. 응용 가능한 예제**

**"검증(Validation) 과정 추가 및 최고 모델 저장"**

학습 데이터(Train set)만으로 학습하면 과적합인지 알 수 없습니다. 별도로 떼어둔 검증 데이터(Validation set)로 매 에폭마다 성능을 평가하고, 검증 성능이 가장 좋을 때의 모델을 저장하는 것이 정석입니다.

```python
best_acc = 0.0
for epoch in range(num_epochs):
    model.train() # 학습 모드
    # ... 학습 루프 ...

    model.eval() # 평가 모드 (Dropout, BatchNorm 등의 동작 변경)
    val_acc = 0.0
    with torch.no_grad(): # 평가 시에는 기울기 계산 불필요 (메모리 절약, 속도 향상)
        for images, labels in val_loader:
            # ... 예측 및 정확도 계산 ...
    
    print(f"Epoch {epoch+1}, Val Acc: {val_acc:.2f}%")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth') # 최고 성능 모델 저장
```

#### **4. 추가하고 싶은 내용 (`model.train()`과 `model.eval()`)**

학습 루프 시작 전에는 `model.train()`, 평가나 추론 전에는 `model.eval()`을 호출하는 습관을 들여야 합니다. Dropout이나 Batch Normalization 같은 층들은 학습할 때와 추론할 때 동작 방식이 다르기 때문입니다. 이를 지키지 않으면 성능이 크게 떨어질 수 있습니다.

---

### **Part 4: 추론(Inference) 및 시각화**

학습이 완료되어 저장된 모델 파일을 불러와서, 학습에 사용되지 않은 새로운 이미지가 어떤 클래스인지 예측하고 그 결과를 눈으로 확인하는 단계입니다. 실제 서비스에 모델을 활용하는 부분입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
# 모델 로드 함수 정의
def load_model(model_path, num_classes):
    # 1. 학습할 때와 동일한 구조의 모델 뼈대를 만듭니다. (가중치는 초기화 상태)
    # 추론만 할 것이므로 pretrained=False로 설정해도 되지만, 구조는 같아야 합니다.
    model = models.resnet50(pretrained=False)
    # 마지막 층도 학습할 때와 똑같이 맞춰줍니다.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 2. 저장된 파라미터(가중치)를 파일에서 불러와 모델 뼈대에 입힙니다.
    # map_location=torch.device("cpu"): GPU로 학습했더라도 CPU 환경에서 불러올 수 있게 매핑합니다.
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    
    # 3. 모델을 평가 모드로 설정합니다. (매우 중요!)
    model.eval()
    return model

# 모델 불러오기 (클래스 개수는 학습 시점의 정보가 필요합니다.)
loaded_model = load_model(model_path, num_classes)
print("✅ 모델이 성공적으로 불러와졌습니다!")

# %%
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 샘플 예측 함수 정의
def predict_sample(image_path, model, class_names):
    # 1. 이미지 불러오기 및 RGB 변환 (PNG 등의 알파 채널 제거)
    image = Image.open(image_path).convert("RGB")
    
    # 2. 학습 시 사용했던 것과 동일한 전처리 적용
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # (C, H, W) -> (1, C, H, W) : 모델은 배치를 입력으로 받으므로 차원 추가 (unsqueeze(0))
    input_tensor = transform(image).unsqueeze(0)

    # 3. 모델 예측
    # with torch.no_grad(): 기울기 계산을 꺼서 메모리 사용량과 연산 시간을 줄입니다.
    with torch.no_grad():
      output = model(input_tensor) # 모델의 출력 (Logits)
      # Softmax를 적용하여 출력을 0~1 사이의 확률 값으로 변환합니다.
      probabilities = F.softmax(output[0], dim=0)
      
      # 가장 높은 확률을 가진 클래스의 인덱스와 그 확률(신뢰도)을 구합니다.
      predicted_idx = torch.argmax(probabilities).item()
      confidence = probabilities[predicted_idx].item()

    # 4. 결과 시각화 및 출력
    plt.imshow(Image.open(image_path)) # 원본 이미지 표시
    plt.title(f"Predicted: {class_names[predicted_idx]} ({confidence * 100:.2f}%)")
    plt.axis('off') # 축 제거
    plt.show()

    print(f"🔍 예측된 클래스 이름: {class_names[predicted_idx]}")
    print(f"📊 확신도(Accuracy): {confidence * 100:.2f}%")

# 테스트할 이미지 경로 입력 및 실행
sample_image = "/content/drive/MyDrive/busanit501-1234/sample.png"
# CPU에서 실행 (loaded_model은 위에서 CPU로 로드됨)
predict_sample(sample_image, loaded_model, class_names)
```

#### **2. 해당 설명**

추론 단계에서 가장 중요한 점 두 가지는 다음과 같습니다.
1.  **학습 시점과 동일한 환경:** 모델 구조(`model.fc` 변경 등)와 입력 데이터의 전처리(`resize`, `normalize`) 과정이 학습할 때와 완전히 동일해야 합니다. 다르다면 모델은 엉뚱한 결과를 내놓을 것입니다.
2.  **`model.eval()`과 `torch.no_grad()`:** 이 두 가지는 세트처럼 사용됩니다. 평가 모드로 전환하여 모델 동작을 추론용으로 바꾸고, 불필요한 기울기 계산을 꺼서 효율성을 극대화합니다.

모델의 최종 출력값(Logits)은 확률이 아닙니다. 이를 사람이 해석하기 쉬운 확률 값으로 바꾸기 위해 **소프트맥스(Softmax)** 함수를 사용합니다. 그리고 `argmax`를 이용해 가장 확률이 높은 클래스를 최종 예측 결과로 선택합니다.

#### **3. 응용 가능한 예제**

**"Top-K 예측 결과 보기"**

가장 높은 확률 하나만 보는 것이 아니라, 상위 3개, 5개의 예측 결과를 보고 싶을 때 `torch.topk`를 사용합니다.
```python
# 상위 5개 확률과 인덱스 추출
top5_prob, top5_idx = torch.topk(probabilities, 5)
for i in range(5):
    class_name = class_names[top5_idx[i].item()]
    prob = top5_prob[i].item()
    print(f"{i+1}순위: {class_name} ({prob*100:.2f}%)")
```
이를 통해 모델이 정답 외에 어떤 클래스와 헷갈려하는지 분석할 수 있습니다.

#### **4. 텍스트 파일 내용 요약 (ResNet 개념)**

텍스트 파일은 ResNet이 **'잔차 연결(Skip Connection, $F(X) + X$)'**을 통해 깊은 신경망의 **기울기 소실 문제**를 해결했다는 핵심을 에스컬레이터 비유를 들어 쉽게 설명하고 있습니다. 입력 정보($X$)가 변형 없이 지름길(shortcut)을 통해 뒤쪽 층으로 직접 전달되므로, 네트워크는 입력과 출력의 차이(잔차, Residual, $F(X)$)만을 학습하면 되어 학습 부담이 줄어들고 안정적으로 깊은 층을 쌓을 수 있게 되었습니다.

---

이 설명이 작성하신 코드와 ResNet의 개념을 이해하고 다른 사람에게 설명하는 데 큰 도움이 되기를 바랍니다. 더 궁금한 점이 있거나 특정 부분에 대해 더 깊은 설명이 필요하시다면 언제든지 말씀해 주세요.