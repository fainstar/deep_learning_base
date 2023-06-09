# 載入套件
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np

# 參數
epochs = 30
lr = 0.04
batch_size = 200
accumulation_steps = 2

# 使用cuda cores
device = torch.device("cuda")
print(device)

# 啟用 cuDNN
torch.backends.cudnn.enabled = True

# 尋找最佳 cuDNN 算法
torch.backends.cudnn.benchmark = True

# 資料轉換
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)

# 載入資料集
train_ds = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
test_ds = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)

# 分割訓練集和驗證集
train_size = int(0.8 * len(train_ds))
val_size = len(train_ds) - train_size
train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

# 更新資料加載器
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

# 撰寫模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2)
        )
  
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 64, 5, 1, 2),
            nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 256, 5, 1, 2),
            nn.ReLU(True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 1024, 5, 1, 2),
            nn.ReLU(True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(1024, 2048, 5, 1, 2),
            nn.ReLU(True),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(2048, 1024, 5, 1, 2),
            #nn.ReLU(True),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(262144, 512),
            nn.ReLU(True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True)
        )

        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.out(x)
        return output


# 驗證集    
def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)
    return val_loss, val_accuracy


# 訓練
def train(model, device, train_loader, criterion, optimizer, scaler, epoch, accumulation_steps):
    model.train()
    optimizer.zero_grad()  
    loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)
        loss = loss / accumulation_steps  
        scaler.scale(loss).backward()  
        if (batch_idx + 1) % accumulation_steps == 0: 
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() 
        if (batch_idx+1) % 10 == 0:
            loss_list.append(loss.item())
            batch = (batch_idx+1) * len(data)
            data_count = len(train_loader.dataset)
            percentage = (100. * (batch_idx+1) / len(train_loader))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ' +
                  f'({percentage:.0f} %)  Loss: {loss.item():.6f}')
    return loss_list

# 測試
def test(model, device, test_loader, loss_list, val_loss_list, val_accuracy_list, lr_list, start_time, end_time):
    model.eval()
    test_loss = 0
    correct = 0
    fig = plt.figure(figsize=(25, 24))  # Adjusted figure size
    gs = gridspec.GridSpec(7, 5)  # Adjusted GridSpec
    # 绘制训练过程中的损失变化
    ax_train_loss = plt.subplot(gs[0, :])
    ax_train_loss.plot(loss_list, 'r', label='Training Loss')
    ax_train_loss.set_title('Training Loss during training')
    ax_train_loss.legend()
    # 绘制验证过程中的损失变化
    ax_val_loss = plt.subplot(gs[1, :])
    ax_val_loss.plot(val_loss_list, 'b', label='Validation Loss')
    ax_val_loss.set_title('Validation Loss during training')
    ax_val_loss.legend()
    # 绘制训练过程中的验证准确率变化
    ax_accuracy = plt.subplot(gs[2, :])
    ax_accuracy.plot(val_accuracy_list, 'g', label='Validation Accuracy')
    ax_accuracy.set_title('Accuracy during training')
    ax_accuracy.legend()
    # 绘制训练过程中的学习率变化
    ax_lr = plt.subplot(gs[3, :])
    ax_lr.plot(lr_list, 'y', label='Learning Rate')
    ax_lr.set_title('Learning Rate during training')
    ax_lr.legend()
    # 绘制前10张测试图片
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            if i == 0:
                for idx in range(2):
                    for j in range(5):
                        ax = plt.subplot(gs[idx+4, j], xticks=[], yticks=[])  # Adjusted indices
                        img = data[idx*5+j].cpu().numpy().transpose((1, 2, 0))
                        img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
                        img = img.clip(0, 1)
                        ax.imshow(img)
                        ax.set_title("{} ({})".format(classes[predicted[idx*5+j].item()], classes[target[idx*5+j].item()]), 
                                    color=("green" if predicted[idx*5+j]==target[idx*5+j] else "red"))
    data_count = len(test_loader.dataset)
    percentage = 100. * correct / data_count
    ax_text = plt.subplot(gs[6, :])  # Adjusted indices
    ax_text.text(0.5, 0.5, f'Accuracy: {correct}/{data_count} ({percentage:.2f}%)\ntraining time: {end_time-start_time:.2f} sec', 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax_text.axis('off')
    plt.tight_layout()
    plt.show()
    torch.cuda.empty_cache()

# 建立 GradScaler 對象
scaler = torch.cuda.amp.GradScaler()

# 建立模型
model = Net().to(device)

# 定義損失函數
criterion = nn.CrossEntropyLoss()  

# 設定優化器
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.7)

# 學習率調度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

# 紀錄開始時間
start_time = time.time()

# 訓練模型
lr_list = []
loss_list = []
val_loss_list = []
val_accuracy_list = []
for epoch in range(1, epochs + 1):
    torch.cuda.empty_cache()
    loss_list += train(model, device, train_loader, criterion, optimizer, scaler, epoch, accumulation_steps)  # 增加參數
    scheduler.step() 
    lr_list.append(optimizer.param_groups[0]['lr']) 
    val_loss, val_accuracy = validate(model, device, val_loader, criterion)
    print(f'Epoch {epoch}: validation loss: {val_loss}, validation accuracy: {val_accuracy}')  
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_accuracy)

# 紀錄結束時間
end_time = time.time()

# 測試模型
test(model, device, test_loader, loss_list, val_loss_list, val_accuracy_list,lr_list, start_time, end_time)
