#載入套件
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

#判斷是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 資料轉換
transform = transforms.Compose(
    [transforms.ToTensor(),
     # 讀入圖像範圍介於[0, 1]之間，將之轉換為 [-1, 1]
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     # ImageNet測出的最佳值
     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

# 批量
batch_size = 1000

# 載入資料集
train_ds = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
test_ds = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)

print(train_ds.data.shape,test_ds.data.shape)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

#撰寫模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 顏色要放在第1維，3:RGB三顏色
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#訓練
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        #每一萬筆資料顯示一次
        if (batch_idx+1) % 10 == 0:
            loss_list.append(loss.item())
            batch = (batch_idx+1) * len(data)
            data_count = len(train_loader.dataset)
            percentage = (100. * (batch_idx+1) / len(train_loader))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ' +
                  f'({percentage:.0f} %)  Loss: {loss.item():.6f}')
    return loss_list

#測試
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()


    # 顯示測試結果
    data_count = len(test_loader.dataset)
    percentage = 100. * correct / data_count
    print(f'準確率: {correct}/{data_count} ({percentage:.2f}%)')


epochs = 10
lr=0.1

# 建立模型
model = Net().to(device)

# 定義損失函數
# 注意，nn.CrossEntropyLoss是類別，要先建立物件，要加 ()，其他損失函數不需要
criterion = nn.CrossEntropyLoss() # F.nll_loss

# 設定優化器(optimizer)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

#訓練模型
loss_list = []
for epoch in range(1, epochs + 1):
    loss_list += train(model, device, train_loader, criterion, optimizer, epoch)

# 對訓練過程的損失繪圖
import matplotlib.pyplot as plt

plt.plot(loss_list, 'r')
plt.show()


#測試模型
test(model, device, test_loader)
