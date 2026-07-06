import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, mean_squared_error


# ==========================================
# 1. 数据集对象（保持原样，高内聚）
# ==========================================
class PECMultiPriorDataset(Dataset):
    def __init__(self, json_path):
        import json
        with open(json_path, 'r') as f:
            raw_data = json.load(f)

        self.waveforms = []  # 356维的时序波形
        self.phys_features = []  # 10维的重要物理先验 (4 + 4 + 2)
        self.labels = []  # 目标：5分类的厚度标签

        for key, value in raw_data.items():
            meta = value['metaDataAndLabel']

            # 1. 波形提取与归一化 (保持无损等比例缩放)
            x_wave = np.array(value['data'], dtype=np.float32)
            wave_min, wave_max = x_wave.min(), x_wave.max()
            if wave_max > wave_min:
                x_wave = (x_wave - wave_min) / (wave_max - wave_min)

            # 2. 提取并拼接真正重要的 3 个物理变量（全部为one-hot列表）
            insulation_oh = meta['InsulationLabel']  # 4维
            liftoff_oh = meta['Lift-offLabel']  # 4维
            weather_jacket_oh = meta['WeatherJacketLabel']  # 2维

            # 横向拼接成一个 10 维的特征向量
            x_phys = insulation_oh + liftoff_oh + weather_jacket_oh

            # 3. 提取 5分类 厚度目标
            y_one_hot = meta['ThicknessLabel']
            y_index = np.argmax(y_one_hot)

            self.waveforms.append(x_wave)
            self.phys_features.append(x_phys)
            self.labels.append(y_index)

        # 转换为 Tensor，预先经过 numpy 包装确保速度
        self.waveforms = torch.tensor(np.array(self.waveforms), dtype=torch.float32)
        self.phys_features = torch.tensor(np.array(self.phys_features), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        # 💡 自动把这套数据集的真实参数暴露给外层的神经网络
        self.wave_dim = self.waveforms.shape[1]  # 应该是 356
        self.phys_dim = self.phys_features.shape[1]  # 4 + 4 + 2 = 10 维
        self.num_classes = len(y_one_hot)  # 5分类

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.waveforms[idx], self.phys_features[idx], self.labels[idx]

# ==========================================
# 1D-CNN
# ==========================================
class PECModel:
    # 💡 隐患修正：加入 wave_len=356，网络根据实际波形采样点动态计算展平特征数
    def __init__(self, num_classes=5, phys_dim=10, wave_len=356, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        ).to(self.device)

        # 💡 [关键改动]：不要写死 32 * 89。
        # 经过两层 MaxPool1d(2, 2)，序列长度缩减为原来的 1/4
        flatten_wave_dim = 32 * (wave_len // 4)

        self.classifier = nn.Sequential(
            nn.Linear(flatten_wave_dim + phys_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            list(self.cnn.parameters()) + list(self.classifier.parameters()),
            lr=lr
        )

    def _forward(self, x_wave, x_phys):
        """内部前向传播方法"""
        x_wave = x_wave.unsqueeze(1)
        x_wave_feat = self.cnn(x_wave)
        x_wave_feat = torch.flatten(x_wave_feat, 1)
        x_combined = torch.cat((x_wave_feat, x_phys), dim=1)
        return self.classifier(x_combined)

    def fit(self, train_loader, epochs=20):
        """训练方法"""
        for epoch in range(epochs):
            self.cnn.train()
            self.classifier.train()
            running_loss = 0.0

            for batch_wave, batch_phys, batch_y in train_loader:
                batch_wave = batch_wave.to(self.device)
                batch_phys = batch_phys.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self._forward(batch_wave, batch_phys)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    def evaluate(self, test_loader):
        """评估方法"""
        self.cnn.eval()
        self.classifier.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_wave, batch_phys, batch_y in test_loader:
                batch_wave = batch_wave.to(self.device)
                batch_phys = batch_phys.to(self.device)

                outputs = self._forward(batch_wave, batch_phys)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # 计算指标
        accuracy = (all_preds == all_targets).mean() * 100
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        mse = mean_squared_error(all_targets, all_preds)

        print(f"\n" + "=" * 40)
        print(f"        --- 融合模型全面评估报告 ---")
        print(f"=" * 40)
        print(f"📈 测试集准确率 (Accuracy)   : {accuracy:.2f}%")
        print(f"🎯 查准率 (Macro Precision) : {precision:.2f}%")
        print(f"🔍 查全率 (Macro Recall)    : {recall:.2f}%")
        print(f"📉 类别均方误差 (Class MSE) : {mse:.4f}")
        print(f"=" * 40)
#Simple MLP
class PECSimpleMLPModel:
    # 💡 默认参数无缝同步为新数据集特征：5分类、10维物理先验
    def __init__(self, wave_dim=356, phys_dim=10, num_classes=5, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 没有任何卷积，输入直接是 356 (波形) + 10 (新物理先验组合) = 366 维
        input_dim = wave_dim + phys_dim

        # 经典感知机架构
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=lr)

    def _forward(self, x_wave, x_phys):
        """内部前向传播：直接拼接原始特征"""
        # x_wave: [Batch, 356], x_phys: [Batch, 10]
        # 顺着特征维拼成 [Batch, 366]
        x_combined = torch.cat((x_wave, x_phys), dim=1)
        return self.mlp(x_combined)

    def fit(self, train_loader, epochs=20):
        """训练方法"""
        for epoch in range(epochs):
            self.mlp.train()
            running_loss = 0.0

            for batch_wave, batch_phys, batch_y in train_loader:
                batch_wave = batch_wave.to(self.device)
                batch_phys = batch_phys.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self._forward(batch_wave, batch_phys)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    def evaluate(self, test_loader):
        """评估方法"""
        self.mlp.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_wave, batch_phys, batch_y in test_loader:
                batch_wave = batch_wave.to(self.device)
                batch_phys = batch_phys.to(self.device)

                outputs = self._forward(batch_wave, batch_phys)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # 指标计算
        accuracy = (all_preds == all_targets).mean() * 100
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        mse = mean_squared_error(all_targets, all_preds)

        print(f"\n" + "=" * 40)
        print(f"    --- 简单 MLP 模型全面评估报告 ---")
        print(f"=" * 40)
        print(f"📈 测试集准确率 (Accuracy)   : {accuracy:.2f}%")
        print(f"🎯 查准率 (Macro Precision) : {precision:.2f}%")
        print(f"🔍 查全率 (Macro Recall)    : {recall:.2f}%")
        print(f"📉 类别均方误差 (Class MSE) : {mse:.4f}")
        print(f"=" * 40)

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super(ResBlock1D, self).__init__()
        # 保持输入输出通道数和长度完全一致的卷积路径
        self.conv_path = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # 💡 核心：残差快车道！输出 = 卷积提取出来的特征 + 原始输入 x
        return self.relu(self.conv_path(x) + x)


# ==========================================
# 1D-ResNet 融合大模型（自适应升级版）
# ==========================================
class PECResNetModel:
    # 💡 升级点：引入 wave_len=356，并同步最新数据集默认值（5分类、10维先验）
    def __init__(self, num_classes=5, phys_dim=10, wave_len=356, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 基础特征映射：先把 1 通道的波形提升到 32 通道
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)  # 第一次池化：wave_len -> wave_len // 2
        ).to(self.device)

        # 串联两个残差块，进行深层特征挖掘而不用担心梯度消失
        self.res_blocks = nn.Sequential(
            ResBlock1D(channels=32),
            ResBlock1D(channels=32)
        ).to(self.device)

        # 进一步下采样，减少全连接层参数量
        self.downsample = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)  # 第二次池化：(wave_len // 2) -> wave_len // 4
        ).to(self.device)

        # 💡 [破除硬编码]：经过两次池化，时序特征长度严格变为原来的 1/4
        # 展平后的总维度 = 64 通道 * (波形长度 // 4)
        flatten_wave_dim = 64 * (wave_len // 4)

        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(flatten_wave_dim + phys_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        # 搜集所有组件的参数进行优化
        all_params = (list(self.init_conv.parameters()) +
                      list(self.res_blocks.parameters()) +
                      list(self.downsample.parameters()) +
                      list(self.classifier.parameters()))
        self.optimizer = optim.Adam(all_params, lr=lr)

    def _forward(self, x_wave, x_phys):
        """内部前向传播"""
        x_wave = x_wave.unsqueeze(1)      # [Batch, 1, wave_len]
        x_feat = self.init_conv(x_wave)   # [Batch, 32, wave_len // 2]
        x_feat = self.res_blocks(x_feat)  # [Batch, 32, wave_len // 2] (残差演进)
        x_feat = self.downsample(x_feat)  # [Batch, 64, wave_len // 4]
        x_feat = torch.flatten(x_feat, 1) # 动态展平

        # 特征拼接：结合一维空间深层特征与环境多物理先验
        x_combined = torch.cat((x_feat, x_phys), dim=1)
        return self.classifier(x_combined)

    def fit(self, train_loader, epochs=20):
        """训练方法"""
        for epoch in range(epochs):
            self.init_conv.train()
            self.res_blocks.train()
            self.downsample.train()
            self.classifier.train()
            running_loss = 0.0

            for batch_wave, batch_phys, batch_y in train_loader:
                batch_wave = batch_wave.to(self.device)
                batch_phys = batch_phys.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self._forward(batch_wave, batch_phys)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    def evaluate(self, test_loader):
        """评估方法"""
        self.init_conv.eval()
        self.res_blocks.eval()
        self.downsample.eval()
        self.classifier.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch_wave, batch_phys, batch_y in test_loader:
                batch_wave = batch_wave.to(self.device)
                batch_phys = batch_phys.to(self.device)

                outputs = self._forward(batch_wave, batch_phys)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        all_preds, all_targets = np.array(all_preds), np.array(all_targets)

        accuracy = (all_preds == all_targets).mean() * 100
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        mse = mean_squared_error(all_targets, all_preds)

        print(f"\n" + "=" * 40)
        print(f"    --- 1D-ResNet 残差模型评估报告 ---")
        print(f"=" * 40)
        print(f"📈 测试集准确率 (Accuracy)   : {accuracy:.2f}%")
        print(f"🎯 查准率 (Macro Precision) : {precision:.2f}%")
        print(f"🔍 查全率 (Macro Recall)    : {recall:.2f}%")
        print(f"📉 类别均方误差 (Class MSE) : {mse:.4f}")
        print(f"=" * 40)
# ==========================================

# ==========================================
if __name__ == "__main__":
    #json_file_path = "D:\\DataSets\\InsideMachine\\PECdataset\\aluminum.json"
    json_file_path = "D:\\DataSets\\InsideMachine\\PECdataset\\S355mildsteel.json"
    dataset = PECMultiPriorDataset(json_file_path)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 实例化大模型类
    # model = PECModel(
    #     num_classes=dataset.num_classes,  # 传入 5
    #     phys_dim=dataset.phys_dim,        # 传入 10
    #     wave_len=dataset.wave_dim,        # 自动传入 356（或实际读取到的任何长度）
    #     lr=0.001
    # )
    # model = PECSimpleMLPModel(
    #     wave_dim=dataset.wave_dim,  # 自动匹配 356
    #     phys_dim=dataset.phys_dim,  # 自动匹配 10
    #     num_classes=dataset.num_classes,  # 自动匹配 5
    #     lr=0.001
    # )
    model = PECResNetModel(
        num_classes=dataset.num_classes,  # 动态传入 5
        phys_dim=dataset.phys_dim,  # 动态传入 10
        wave_len=dataset.wave_dim,  # 动态传入 356
        lr=0.001
    )
    #model=PECSimpleMLPModel(wave_dim=356, phys_dim=5, num_classes=9, lr=0.001)
    #model = PECResNetModel(num_classes=9, phys_dim=5, lr=0.001)
    # 直接串联业务
    model.fit(train_loader, epochs=40)
    model.evaluate(test_loader)