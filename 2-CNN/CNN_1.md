
---

### **코드 종합 설명**

이 코드는 이미지 인식의 핵심 기술인 **CNN(합성곱 신경망, Convolutional Neural Network)**을 사용하여, 사용자가 직접 준비한 이미지 데이터셋을 분류하는 모델을 처음부터 끝까지 구축하는 과정을 보여주는 미니 프로젝트입니다.

1.  **데이터 준비 및 로딩:** 구글 드라이브에 저장된 이미지 폴더로부터 데이터를 불러옵니다. `torchvision.datasets.ImageFolder`를 사용하여 폴더 구조 자체를 데이터셋의 레이블로 자동 인식하게 하고, `transforms`를 통해 모든 이미지의 크기를 통일하고 텐서로 변환하며 **정규화(Normalization)**하는 전처리 과정을 수행합니다.
2.  **CNN 모델 정의:** `nn.Module`을 상속받아 두 개의 합성곱 계층(`nn.Conv2d`), 활성화 함수(`nn.ReLU`), 풀링 계층(`nn.MaxPool2d`) 그리고 두 개의 완전 연결 계층(`nn.Linear`)으로 구성된 간단하면서도 표준적인 CNN 모델을 정의합니다.
3.  **모델 학습 및 저장:** GPU 사용이 가능하도록 설정한 뒤, 정의된 모델을 학습시킵니다. `CrossEntropyLoss`와 `Adam` 옵티마이저를 사용하여 훈련 루프(Training Loop)를 실행하고, 학습이 완료된 모델의 가중치(`state_dict`)를 파일로 저장하여 재사용할 수 있도록 합니다.
4.  **모델 추론 및 예측:** 저장된 모델 가중치를 다시 불러와 평가 모드(`model.eval()`)로 전환한 뒤, 완전히 새로운 샘플 이미지를 입력받아 어떤 클래스에 속하는지 예측하고 그 결과를 시각화하는 실용적인 추론 함수를 구현합니다.

이 코드를 통해 이미지 데이터에 특화된 CNN의 핵심 구성 요소(Conv, ReLU, Pool, FC)가 어떻게 유기적으로 결합하여 이미지의 특징을 추출하고 분류를 수행하는지, 그리고 실제 이미지 분류 프로젝트의 전체 워크플로우를 명확하게 이해할 수 있습니다.

---

### **Part 1: 데이터 준비 - 이미지 폴더를 데이터셋으로 변환하기**

컴퓨터에 저장된 이미지 파일들을 PyTorch가 이해할 수 있는 데이터셋과 데이터로더 형태로 변환하는 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 📌 1. 데이터가 저장된 루트 폴더 경로 설정
data_dir = "/content/drive/MyDrive/busanit501-1234/class/"
# 이 폴더 안에는 'cat', 'dog', 'tiger'와 같이 각 클래스별로 이미지가 담긴 하위 폴더가 있어야 합니다.

# 📌 2. 이미지 전처리(Transform) 파이프라인 정의
transform = transforms.Compose([
    # Resize: 모든 이미지의 크기를 224x224 픽셀로 강제 조정합니다.
    # 모델에 입력되는 이미지 크기를 통일하기 위함입니다.
    transforms.Resize((224, 224)),
    # ToTensor: PIL 이미지나 Numpy 배열을 PyTorch 텐서로 변환합니다.
    # 픽셀 값을 [0, 255]에서 [0.0, 1.0] 범위로 자동으로 스케일링하고,
    # 차원 순서를 (H, W, C) -> (C, H, W)로 변경합니다.
    transforms.ToTensor(),
    # Normalize: 텐서의 픽셀 값을 정규화합니다.
    # 각 채널에 대해 (pixel - mean) / std 연산을 수행합니다.
    # [-1, 1] 범위로 정규화하여 학습을 안정시킵니다.
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 📌 3. ImageFolder로 데이터셋 로드
# ImageFolder는 지정된 폴더 구조를 자동으로 인식하여 데이터셋을 만듭니다.
# 예를 들어, 'class/cat' 폴더의 이미지는 'cat'이라는 레이블을,
# 'class/dog' 폴더의 이미지는 'dog' 레이블을 자동으로 부여받습니다.
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 📌 4. DataLoader 생성
# shuffle=True로 설정하여 학습 시 데이터 순서를 무작위로 섞습니다.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 📌 5. 클래스 목록 확인
# ImageFolder는 폴더 이름을 기준으로 클래스 목록을 자동으로 생성합니다.
class_names = train_dataset.classes
print(f"✅ 클래스 목록: {class_names}") # 결과 예시: ['cat', 'dog', 'tiger']
```

#### **2. 해당 설명**

이 파트는 CNN 프로젝트의 첫 단추인 **데이터 파이프라인 구축**을 보여줍니다. `transforms.Compose`는 이미지에 대한 일련의 전처리 과정을 정의하는 파이프라인입니다. 모든 이미지를 동일한 크기로 만들고(`Resize`), PyTorch가 다룰 수 있는 텐서 형태로 바꾸며(`ToTensor`), 학습을 안정시키는 **정규화(`Normalize`)**를 수행하는 것은 거의 모든 이미지 분류 프로젝트의 표준적인 절차입니다.

**`datasets.ImageFolder`**는 PyTorch의 매우 편리한 기능입니다. 우리가 직접 데이터 경로와 레이블을 일일이 매핑할 필요 없이, **'클래스 이름으로 된 폴더 안에 해당 클래스의 이미지를 넣는다'** 는 간단한 규칙만 지키면 자동으로 데이터셋을 구성해줍니다. 이는 코드의 복잡성을 크게 줄여주고 데이터 관리를 직관적으로 만들어줍니다.

#### **3. 응용 가능한 예제**

**"의료 영상(X-ray) 분류 데이터셋 구축"**

`/medical_images/` 폴더 안에 `/normal/` 과 `/pneumonia/` 라는 두 개의 하위 폴더를 만들고 각각의 X-ray 이미지를 넣어두기만 하면, `datasets.ImageFolder(root='/medical_images/', ...)` 코드 한 줄로 '정상'과 '폐렴'을 분류하는 데이터셋을 즉시 만들 수 있습니다.

#### **4. 추가하고 싶은 내용 (데이터 증강 추가)**

모델의 성능을 더 높이기 위해, `transform` 파이프라인에 데이터 증강(Data Augmentation) 단계를 추가할 수 있습니다. 이는 모델이 더 다양한 형태의 이미지를 학습하여 일반화 성능을 높이는 데 큰 도움이 됩니다.

```python
transform_augmented = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 좌우 반전
    transforms.RandomRotation(10),       # 10도 내외로 무작위 회전
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

#### **5. 심화 내용 (Normalize의 mean과 std 값)**

예제에서는 `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]`를 사용하여 픽셀 범위를 `[-1, 1]`로 정규화했습니다. 하지만 ImageNet과 같이 거대한 데이터셋으로 사전 학습된 모델을 사용하는 **전이 학습(Transfer Learning)**의 경우, 해당 모델이 학습될 때 사용했던 ImageNet 데이터셋의 평균(`[0.485, 0.456, 0.406]`)과 표준편차(`[0.229, 0.224, 0.225]`)를 그대로 사용해야 최상의 성능을 얻을 수 있습니다.

---

### **Part 2: CNN 모델 정의 및 학습**

"우편물 분류 비유"에서 설명된 합성곱, 활성화, 풀링, 완전 연결 계층을 코드로 구현하고, 실제 데이터로 모델을 훈련시키는 과정입니다.

#### **1. 코드, 문법 및 개별 설명**

```python
# %%
import torch.nn as nn
import torch.optim as optim

# 📌 1. CNN 모델 클래스 정의
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # --- 특징 추출기 (Feature Extractor) ---
        # 1. 첫 번째 합성곱 계층 (Convolution)
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # - in_channels=3: 입력 이미지의 채널 수 (RGB 이미지이므로 3)
        # - out_channels=32: 사용할 필터(커널)의 개수. 출력 특징 맵(feature map)의 깊이가 됨.
        # - kernel_size=3: 3x3 크기의 필터를 사용
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # 2. 두 번째 합성곱 계층
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 3. 맥스 풀링 계층 (Max Pooling)
        # nn.MaxPool2d(kernel_size, stride)
        # 2x2 영역에서 가장 큰 값만 남기고 나머지는 버려, 특징 맵의 크기를 절반으로 줄임.
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU() # 4. 활성화 함수 (Activation)

        # --- 분류기 (Classifier) ---
        # 5. 첫 번째 완전 연결 계층 (Fully Connected Layer)
        # 입력 크기 계산:
        # 초기 이미지: 224x224
        # pool1 후: 112x112
        # pool2 후: 56x56
        # 최종 특징 맵의 채널 수는 conv2의 out_channels인 64.
        # 따라서 펼친(flatten) 후의 크기는 64 * 56 * 56
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        # 6. 두 번째 (최종) 완전 연결 계층
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x): # 데이터의 흐름 정의
        # 입력 shape: (Batch, 3, 224, 224)
        x = self.pool(self.relu(self.conv1(x))) # Conv1 -> ReLU -> Pool
        # shape: (Batch, 32, 112, 112)
        x = self.pool(self.relu(self.conv2(x))) # Conv2 -> ReLU -> Pool
        # shape: (Batch, 64, 56, 56)

        # .view()를 이용해 4D 텐서를 2D 텐서로 펼침 (Flatten)
        x = x.view(-1, 64 * 56 * 56)
        # shape: (Batch, 200704)
        x = self.relu(self.fc1(x)) # FC1 -> ReLU
        # shape: (Batch, 512)
        x = self.fc2(x) # 최종 출력 (각 클래스에 대한 점수, Logits)
        # shape: (Batch, num_classes)
        return x

# 📌 2. 모델, 손실함수, 옵티마이저 초기화
num_classes = len(class_names)
model = CustomCNN(num_classes)
criterion = nn.CrossEntropyLoss() # 다중 클래스 분류를 위한 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ... (학습 루프 및 모델 저장 코드 생략) ...
# 학습 루프는 이전 MLP 예제와 거의 동일하며, GPU 사용 코드(to(device))가 추가됨
```

#### **2. 해당 설명**

이 파트는 **"우편물 분류 비유"**를 코드로 완벽하게 구현한 것입니다.

*   **`conv1`, `conv2` (합성곱 연산):** 이미지 위를 3x3 크기의 '돋보기'(필터)가 이동하면서 수직선, 수평선, 특정 색상 등 저수준의 시각적 특징을 찾아냅니다. 이는 우편물에서 '우편번호'나 '주소' 같은 핵심 정보를 찾아내는 과정과 같습니다.
*   **`relu` (활성화 함수):** 찾아낸 특징 중 의미 있는 양수 값만 남기고 음수 값은 0으로 만들어, 불필요한 노이즈를 제거하고 중요한 특징을 강조합니다.
*   **`pool` (풀링 연산):** 특징 맵의 크기를 줄여 계산량을 감소시키고, 위치 변화에 약간 둔감한(강건한) 특징을 만듭니다. '서울특별시 강남구'를 '강남'으로 요약하는 것과 같습니다.
*   **`x.view(-1, ...)` (Flatten):** 2차원 형태의 특징 맵(이미지)을 1차원의 긴 벡터로 펼쳐서, 전통적인 신경망(완전 연결 계층)에 입력할 수 있도록 준비하는 과정입니다.
*   **`fc1`, `fc2` (완전연결층):** 추출된 모든 시각적 특징들을 종합하여, 최종적으로 이 이미지가 어떤 클래스에 속할 확률이 가장 높은지 판단합니다. 이는 우편번호와 주소를 보고 '이 우편물은 강남우체국으로!'라고 최종 결정하는 것과 같습니다.

#### **3. 응용 가능한 예제**

**"더 깊은 CNN 모델 만들기 (VGG 스타일)"**

더 많은 합성곱 계층을 쌓아 모델의 표현력을 높일 수 있습니다. 예를 들어, `(Conv-Conv-Pool)` 블록을 여러 개 반복하여 VGGNet과 유사한 구조의 깊은 모델을 만들 수 있습니다.

```python
# ...
self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
# forward에서는 self.pool(self.relu(self.conv4(self.relu(self.conv3(x))))) 와 같이 추가
# fc1의 입력 크기도 풀링이 한 번 더 적용되었으므로 64*56*56이 아닌 128*28*28로 변경해야 함
```

#### **4. 추가하고 싶은 내용 (GPU 사용)**

이미지 데이터와 CNN 모델은 계산량이 매우 크기 때문에 CPU만으로는 학습에 수 시간에서 수 일이 걸릴 수 있습니다. `device = torch.device(...)`와 `model.to(device)`, `images.to(device)` 코드는 모델과 데이터를 GPU로 보내어 병렬 처리를 통해 학습 속도를 수십 배 이상 향상시키는 필수적인 과정입니다.

#### **5. 심화 내용 (FC층 입력 크기 자동 계산)**

`self.fc1 = nn.Linear(64 * 56 * 56, 512)` 에서 `64 * 56 * 56`이라는 숫자를 직접 계산하는 것은 번거롭고 이미지 크기가 바뀌면 에러를 유발합니다. 이를 해결하기 위해, `__init__`에서 합성곱 부분만 정의한 뒤, 가짜 데이터(`torch.randn(1, 3, 224, 224)`)를 한 번 통과시켜 나온 출력의 크기를 보고 FC층의 입력 크기를 동적으로 결정하는 기법을 사용하면 훨씬 유연한 모델을 만들 수 있습니다.

---

### **Part 3: 모델 추론 - 새로운 이미지 예측하기**

학습이 완료된 모델을 사용하여, 세상에 존재하는 새로운 이미지를 분류하는 실용적인 예측 과정을 구현합니다.

*이 파트는 이전 회귀 예제의 예측 함수와 매우 유사하며, 이미지 데이터에 맞게 특화되어 있습니다. 코드 설명은 위 코드 블록에 포함되어 있으므로, 개념적 설명과 심화 내용에 집중하겠습니다.*

#### **2. 해당 설명**

**추론(Inference)**은 학습된 모델을 실제 문제 해결에 사용하는 단계입니다. 이 과정은 새로운 이미지를 입력받았을 때, **학습 시 사용했던 전처리(`transform`) 과정을 '똑같이' 적용**하는 것이 매우 중요합니다. 모델은 특정 크기와 특정 정규화 분포를 가진 데이터에 맞춰 학습되었기 때문에, 새로운 데이터도 동일한 '언어'로 번역해주어야 올바르게 이해하고 예측할 수 있습니다. `predict_sample` 함수는 바로 이 표준적인 추론 파이프라인(이미지 로드 → 전처리 → 배치 차원 추가 → 모델 예측 → 결과 해석)을 보여줍니다.

#### **3. 응용 가능한 예제**

**"실시간 웹캠 영상 객체 탐지"**

웹캠에서 매 프레임마다 이미지를 받아와, 이 `predict_sample` 함수와 유사한 로직을 통과시키면 실시간으로 영상 속의 객체가 무엇인지 분류하는 어플리케이션을 만들 수 있습니다.

#### **4. 추가하고 싶은 내용 (Softmax로 확률 얻기)**

현재 모델의 최종 출력(`output`)은 각 클래스에 대한 점수(logit)입니다. `torch.max`를 사용해 가장 높은 점수를 가진 클래스를 예측하는 것만으로도 충분하지만, 각 클래스에 대한 '확신도(확률)'를 보고 싶을 때가 있습니다. 이때 **소프트맥스(Softmax)** 함수를 사용합니다.

```python
import torch.nn.functional as F

with torch.no_grad():
    output = model(image)
    probabilities = F.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

print(f"예측 클래스: {class_names[predicted.item()]}")
print(f"확신도: {confidence.item() * 100:.2f}%")
```

#### **5. 심화 내용 (전이 학습, Transfer Learning)**

이 예제처럼 모델을 '처음부터(from scratch)' 학습시키는 것은 매우 많은 데이터와 시간이 필요합니다. 실제 대부분의 프로젝트에서는 ImageNet과 같이 거대한 데이터셋으로 미리 학습된 **사전 학습 모델(Pre-trained Model)**(예: ResNet, VGG, EfficientNet)을 가져와서, 마지막 분류기 부분(`fc2`)만 우리의 데이터셋에 맞게 교체하여 학습시키는 **전이 학습(Transfer Learning)**을 사용합니다. 이는 훨씬 적은 데이터로도 매우 높은 성능을 빠르고 쉽게 얻을 수 있는 강력한 기법입니다. `torchvision.models`에서 다양한 사전 학습 모델을 쉽게 불러와 사용할 수 있습니다.