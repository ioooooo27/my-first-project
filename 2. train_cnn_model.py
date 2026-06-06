import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = (224,224)
batch_size = 16
epochs = 10
num_classes = 6
class_names = ["apple","banana","orange","grape","strawberry","pear"]

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder("fruit_images_split/train", transform)
val_dataset = datasets.ImageFolder("fruit_images_split/val", transform)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

# CNN教师模型
class FruitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*28*28, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

model = FruitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练
history = {"loss":[], "acc":[], "val_loss":[], "val_acc":[]}
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = torch.max(outputs,1)
        total += labels.size(0)
        correct += (pred==labels).sum().item()
    epoch_loss = running_loss/len(train_loader)
    epoch_acc = correct/total

    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, pred = torch.max(outputs,1)
            val_total += labels.size(0)
            val_correct += (pred==labels).sum().item()
    epoch_val_loss = val_loss/len(val_loader)
    epoch_val_acc = val_correct/val_total

    history["loss"].append(epoch_loss)
    history["acc"].append(epoch_acc)
    history["val_loss"].append(epoch_val_loss)
    history["val_acc"].append(epoch_val_acc)
    print(f"[{epoch+1}/{epochs}] loss:{epoch_loss:.4f} acc:{epoch_acc:.4f} val_loss:{epoch_val_loss:.4f} val_acc:{epoch_val_acc:.4f}")

# 保存模型
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/fruit_classification_model.pth")

# 绘制曲线
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history["loss"], label="train loss")
plt.plot(history["val_loss"], label="val loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history["acc"], label="train acc")
plt.plot(history["val_acc"], label="val acc")
plt.legend()
plt.tight_layout()
plt.savefig("assets/training_history.png")
plt.show()