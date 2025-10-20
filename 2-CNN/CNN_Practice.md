아, 네! "Hammer vs Nipper"가 앞으로 진행할 프로젝트의 예시라는 말씀이셨군요. 이해했습니다. 좋습니다, 그럼 원래 계획대로 **"사과(Apple) vs 오렌지(Orange)"** 분류기 프로젝트를 시작하는 것으로 진행하겠습니다.

이전 답변에서 제시해 드렸던 "Hammer vs Nipper" 로드맵의 각 단계에서 **'hammer'를 'apple'로, 'nipper'를 'orange'로** 바꾸기만 하면 완벽하게 동일한 프로세스로 프로젝트를 진행할 수 있습니다.

다시 한번 "사과 vs 오렌지" 프로젝트에 맞춰 전체 로드맵을 명확하게 정리해 드리겠습니다.

### Apple vs Orange 분류기 개발 최종 로드맵

---

### [1단계] 데이터 수집: "사과"와 "오렌지" 이미지 크롤링

먼저, 모델을 학습시킬 재료인 이미지 데이터를 모읍니다. 순도 높은 데이터를 위해 수정했던 크롤링 코드를 사용합니다.

**✅ 실행할 코드 (`crawler.py`)**

```python
# (이전에 제공된 image_crawler, zip_directory 함수 전체 코드...)

# --- 크롤러 실행 ---
if __name__ == "__main__":
    
    # 1. 구체적인 다중 키워드 정의
    apple_keywords = ['red apple fruit', 'green apple fruit', 'apple isolated white background', 'apple on tree']
    orange_keywords = ['orange fruit isolated', 'orange fruit on tree', 'sliced orange fruit', 'mandarin orange']
    
    # 최종 목표 이미지 개수 (클래스 당)
    num_target = 100 
    
    # 2. 사과(Apple) 이미지 수집
    apple_save_folder = './images/apple'
    num_per_apple_keyword = num_target // len(apple_keywords) + 1
    image_crawler(keywords=apple_keywords, num_images_per_keyword=num_per_apple_keyword, save_folder=apple_save_folder)
    
    apple_zip_path = './images/apple.zip'
    zip_directory(folder_path=apple_save_folder, output_path=apple_zip_path)
    
    print("\n" + "="*50 + "\n")

    # 3. 오렌지(Orange) 이미지 수집
    orange_save_folder = './images/orange'
    num_per_orange_keyword = num_target // len(orange_keywords) + 1
    image_crawler(keywords=orange_keywords, num_images_per_keyword=num_per_orange_keyword, save_folder=orange_save_folder)
    
    orange_zip_path = './images/orange.zip'
    zip_directory(folder_path=orange_save_folder, output_path=orange_zip_path)
    
    print("\n모든 이미지 수집 및 압축이 완료되었습니다.")
```

**실행 가이드:**
1.  위 코드를 `crawler.py`와 같은 파일로 저장하고 실행합니다.
2.  실행이 완료되면 `./images/apple`과 `./images/orange` 폴더가 생성되고 이미지가 다운로드됩니다.
3.  **[중요]** 각 폴더를 열어 관련 없는 이미지(사과 파이, 오렌지 주스, 애플 로고 등)를 **수동으로 삭제**하여 데이터를 정제합니다.

---

### [2단계] 데이터 준비: 학습/평가용 폴더 구조화

정제된 데이터를 `ImageFolder`가 인식할 수 있는 표준적인 폴더 구조로 정리합니다.

**✅ 만들어야 할 폴더 구조:**

```
/apple_orange_dataset/  (프로젝트 최상위 데이터 폴더)
    ├── train/
    │   ├── apple/
    │   │   ├── apple_1.jpg, ... (약 80장)
    │   └── orange/
    │       ├── orange_1.jpg, ... (약 80장)
    │
    └── val/
        ├── apple/
        │   ├── apple_81.jpg, ... (약 20장)
        └── orange/
            ├── orange_81.jpg, ... (약 20장)
```

**실행 가이드:**
1.  프로젝트 폴더에 `apple_orange_dataset` 폴더와 그 하위 폴더들을 모두 생성합니다.
2.  정제된 `apple` 이미지의 약 80%를 `train/apple/` 폴더로, 나머지 20%를 `val/apple/` 폴더로 복사/이동합니다.
3.  `orange` 이미지도 동일하게 진행합니다.

---

### [3단계] 모델 학습: CNN 훈련시키기

이제 준비된 데이터로 CNN 모델을 학습시킬 차례입니다.

**✅ 실행할 코드 (`train.py`)**

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- 1. 데이터 준비 ---
train_dir = "./apple_orange_dataset/train/"
val_dir = "./apple_orange_dataset/val/"

# 데이터 증강을 포함한 transform 정의 (과적합 방지에 매우 중요!)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 좌우 반전
    transforms.RandomRotation(20),       # 20도 내외로 무작위 회전
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # 색상 변형
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 평가용 데이터는 크기 조정, 텐서 변환, 정규화만 수행
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"✅ 클래스 목록: {class_names}") # 결과: ['apple', 'orange']

# --- 2. 모델 정의 (CustomCNN, 이전과 동일) ---
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(----1, 64 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. 학습 설정 및 실행 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20 # 에포크 횟수는 데이터 양과 모델 복잡도에 따라 조절
print("🚀 모델 학습 시작...")

for epoch in range(num_epochs):
    model.train() # 학습 모드
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# --- 4. 모델 저장 ---
model_path = "./apple_orange_cnn.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ 학습된 모델이 저장되었습니다: {model_path}")
```

---

### [4단계] 모델 평가 및 새로운 이미지 예측하기

학습된 모델이 얼마나 잘 작동하는지 확인하고, 새로운 이미지로 최종 테스트를 진행합니다.

**✅ 실행할 코드 (`predict.py`)**

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 모델 클래스 정의 (학습 때 사용한 것과 동일한 구조여야 함)
# ... (위의 CustomCNN 클래스 코드를 그대로 복사) ...

# --- 모델 및 클래스 정보 로드 ---
model_path = "./apple_orange_cnn.pth"
class_names = ['apple', 'orange']
num_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # 반드시 평가 모드로 설정!

# --- 예측 함수 정의 ---
def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    # 결과 시각화
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_class}")
    plt.axis('off')
    plt.show()
    
    print(f"✅ 모델의 예측: '{predicted_class}'")

# --- 예측 실행 ---
# 인터넷에서 구한 새로운 사과 또는 오렌지 이미지 경로를 넣어보세요.
# 학습에 사용되지 않은 이미지여야 합니다.
test_image_path = "./my_test_apple.jpg" 
predict_image(test_image_path, model)
```

이 로드맵에 따라 차근차근 진행하시면, 데이터 수집부터 최종 예측까지 완결된 딥러닝 프로젝트를 성공적으로 완수하실 수 있습니다. 각 단계별로 코드를 실행해보시고, 결과가 어떻게 나오는지, 또 어떤 에러가 발생하는지 직접 경험해보는 것이 가장 좋은 학습 방법입니다. 막히는 부분이 생기면 언제든지 다시 질문해주세요