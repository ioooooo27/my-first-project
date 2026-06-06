import tkinter as tk
import cv2
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = (224,224)
num_classes = 6
class_names = ["apple","banana","orange","grape","strawberry","pear"]

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class FruitCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,3,1,1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32,64,3,1,1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64,128,3,1,1), torch.nn.ReLU(), torch.nn.MaxPool2d(2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128*28*28, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

model = FruitCNN().to(device)
model.load_state_dict(torch.load("app/models/fruit_classification_model.pth", map_location=device))
model.eval()

root = tk.Tk()
root.title("水果分类-摄像头")
root.geometry("700x600")

label_cam = tk.Label(root)
label_cam.pack(pady=10)

label_res = tk.Label(root, text="实时结果：", font=("Arial",16))
label_res.pack(pady=5)

cap = cv2.VideoCapture(0)

def update():
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img_show = img.resize((640,480))
        img_tk = ImageTk.PhotoImage(img_show)
        label_cam.config(image=img_tk)
        label_cam.image = img_tk

        # 预测
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_tensor)
            _, idx = torch.max(out,1)
            cls_name = class_names[idx.item()]
        label_res.config(text=f"实时预测：{cls_name}")
    root.after(10, update)

update()
root.mainloop()
cap.release()