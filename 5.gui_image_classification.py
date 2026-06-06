import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from torchvision import datasets

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
model.load_state_dict(torch.load("models/fruit_classification_model.pth", map_location=device))
model.eval()

root = tk.Tk()
root.title("水果分类-单图")
root.geometry("600x500")

label_img = tk.Label(root)
label_img.pack(pady=10)

label_result = tk.Label(root, text="结果：", font=("Arial",16))
label_result.pack(pady=5)

def open_image():
    path = filedialog.askopenfilename(filetypes=[("Image","*.jpg;*.png")])
    if not path:
        return
    img = Image.open(path).convert("RGB")
    img_show = img.resize((400,400))
    img_tk = ImageTk.PhotoImage(img_show)
    label_img.config(image=img_tk)
    label_img.image = img_tk

    # 预测
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)
        _, idx = torch.max(out,1)
        cls_name = class_names[idx.item()]
    label_result.config(text=f"预测：{cls_name}")

btn = tk.Button(root, text="选择图片并识别", command=open_image, font=("Arial",14))
btn.pack(pady=5)

root.mainloop()