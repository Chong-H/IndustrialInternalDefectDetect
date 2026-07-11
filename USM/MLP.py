import os
import re
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. 动态数据加载与解析模块
# ==========================================
class DynamicUSMProcessor:
    """自适应数据处理器：自动识别任意Meter文件的特征维度与类别总数"""

    def __init__(self, data_path: str, batch_size: int = 16):
        self.data_path = data_path
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.num_features = 0
        self.num_classes = 0

    def load_and_adapt(self, test_size: float = 0.2, random_state: int = 42):
        # 直接用最纯粹的逗号分隔符读取 CSV
        df = pd.read_csv(self.data_path, header=0)

        # 只剔除完全空白的死行，不误杀任何特征列
        df.dropna(how='all', inplace=True)

        # 动态捕捉数据集的物理元数据
        X = df.iloc[:, 1:-1].values
        y = df.iloc[:, -1].astype(int).values

        self.num_features = X.shape[1]
        self.num_classes = len(np.unique(y))

        print(
            f"📊 成功读取 CSV 数据！总样本数: {X.shape[0]} | 特征维度: {self.num_features} | 类别数: {self.num_classes}")

        # 将 UCI 的 1-based 标签转化为 PyTorch 的 0-based 索引
        y = y - 1

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 标准化特征空间
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        return X_train, X_val, y_train, y_val

    def create_loaders(self, X_train, X_val, y_train, y_val):
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        )


# ==========================================
# 2. 通用自适应多分类 MLP
# ==========================================
class UniversalMLP(nn.Module):
    """自适应全连接多分类神经网络"""

    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.25):
        super(UniversalMLP, self).__init__()
        # 针对小样本高维表格数据的经典对称瓶颈拓扑结构
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            #将64列特征各自归一化为μ=0,σ=1的分布，提升训练稳定性
            nn.BatchNorm1d(64),
            #与Relu类似，但在负数区间不再变为0，而是变为原数据的0.1倍，避免神经元死亡
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(32, num_classes)  # 输出层自适应多分类
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 3. 动态模型路由与核心管道
# ==========================================
class DynamicUSMPipeline:
    """动态流水线：支持动态文件命名持久化与根据输入维度自动寻根路由加载"""

    def __init__(self, model_dir: str = "./models", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_dir = model_dir
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        os.makedirs(self.model_dir, exist_ok=True)

    def _generate_model_name(self, num_features: int, num_classes: int) -> str:
        """根据当前的特征结构动态生成唯一的标准化模型文件名"""
        return os.path.join(self.model_dir, f"USM_Features{num_features}_Classes{num_classes}.pt")

    def _auto_route_and_load_model(self, num_features: int) -> nn.Module:
        """
        核心路由机制：在磁盘中扫描符合当前特征输入维度的 .pt 文件，
        自动解析其对应的分类类别数，动态初始化网络并完成权重加载。
        """
        search_pattern = os.path.join(self.model_dir, f"USM_Features{num_features}_Classes*.pt")
        matched_files = glob.glob(search_pattern)

        if not matched_files:
            raise FileNotFoundError(
                f"❌ 错误: 未能在目录 {self.model_dir} 中找到特征维度为 {num_features} 的训练就绪模型文件。请先运行 train。")

        # 选取最新或最匹配的第一个文件
        target_model_path = matched_files[0]

        # 正则表达式逆向解析文件名中的特征和类别数
        match = re.search(r"USM_Features(\d+)_Classes(\d+)\.pt", target_model_path)
        if not match:
            raise ValueError(f"无法从文件名解析元数据: {target_model_path}")

        features_dim = int(match.group(1))
        classes_dim = int(match.group(2))

        print(f"--> [智能路由] 成功匹配权重文件: {os.path.basename(target_model_path)}")
        print(f"--> [智能路由] 自动识别网络架构 -> 特征维度: {features_dim}, 目标分类数: {classes_dim}")

        # 动态实例化网络并加载权重
        model = UniversalMLP(input_dim=features_dim, num_classes=classes_dim)
        model.load_state_dict(torch.load(target_model_path, map_location=self.device))
        return model.to(self.device)

    def train(self, processor: DynamicUSMProcessor, train_loader, val_loader, epochs: int = 40, lr: float = 0.005):
        """完全解耦的训练接口，动态创建网络并导出带元数据的权重文件"""
        # 动态获取当前处理器的维度指标
        num_features = processor.num_features
        num_classes = processor.num_classes

        # 动态实例化模型
        model = UniversalMLP(input_dim=num_features, num_classes=num_classes).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

        save_path = self._generate_model_name(num_features, num_classes)
        best_acc = 0.0

        print(f"\n================ 启动自适应多分类训练 ================")
        print(f"数据指纹 -> 特征列数: {num_features} | 故障类别数: {num_classes}")
        print(f"持久化预定路径: {save_path}\n--------------------------------------------------")

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss, correct = 0.0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)

            train_loss /= len(train_loader.dataset)
            train_acc = correct.double() / len(train_loader.dataset)

            # 伴随验证
            model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += torch.sum(preds == labels.data)

            val_loss /= len(val_loader.dataset)
            val_acc = val_correct.double() / len(val_loader.dataset)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)
                print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} ⭐ 模型已更新保存")
            else:
                if epoch % 10 == 0 or epoch == epochs:
                    print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    def eval(self, val_loader, num_features: int):
        """评估接口：计算多指标分类报告（查准率、查全率、F1-score）"""
        from sklearn.metrics import classification_report, confusion_matrix

        model = self._auto_route_and_load_model(num_features)
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 映射回原始的 1, 2, 3, 4 物理标签，方便人肉眼看
        target_names = [f"Class {i + 1}" for i in sorted(np.unique(all_labels))]

        print("\n==================== 📊 详细评估报告 ====================")
        # 打印查准率、查全率、F1-Score
        print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

        print("==================== 🧱 混淆矩阵 (Confusion Matrix) ====================")
        # 横轴是预测值，纵轴是真实值
        print(confusion_matrix(all_labels, all_preds))
        print("========================================================")

        # 提取出总体的 macro f1 作为返回值（如果外界需要）
        from sklearn.metrics import f1_score
        return f1_score(all_labels, all_preds, average='macro')

    def pred(self, raw_features: np.ndarray, scaler: StandardScaler) -> np.ndarray:
        """预测接口：根据输入的原始特征维度自动加载对应权重，无缝支持单样本或批量多样本推理"""
        # 支持单样本降维展平转换为标准矩阵输入
        if raw_features.ndim == 1:
            raw_features = raw_features.reshape(1, -1)

        num_features = raw_features.shape[1]
        model = self._auto_route_and_load_model(num_features)
        model.eval()

        # 依靠对应数据集在训练期拟合好的 Scaler 进行对齐缩放
        scaled_features = scaler.transform(raw_features)
        tensor_input = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = model(tensor_input)
            _, preds = torch.max(outputs, 1)

        # 加上 1 还原回物理故障标签范围 1, 2, 3, 4
        return preds.cpu().numpy() + 1


# ==========================================
# 4. 多文件自动化流水线执行验证
# ==========================================
# ==========================================
# 4. 多文件自动化流水线执行验证
# ==========================================
if __name__ == "__main__":
    # 基础配置
    DATA_DIR = r"D:\DataSets\InsideMachine\UltrasonicFlowmeterDiagnosticsDataSet"
    meter_c_path = os.path.join(DATA_DIR, "flowmeter_D_mined.csv")

    # 初始化控制流水线
    pipeline = DynamicUSMPipeline(model_dir="./saved_models")

    # ----------------------------------------------------
    # 控制变量：在这里修改任务类型 ("train", "eval", "pred")
    # ----------------------------------------------------
    task = "eval"
    # task = "train"
    # 1. 动态加载解析（所有任务都需要初始化数据结构）
    if os.path.exists(meter_c_path):
        processor_c = DynamicUSMProcessor(data_path=meter_c_path, batch_size=16)
        X_train, X_val, y_train, y_val = processor_c.load_and_adapt(test_size=0.2)
        train_loader, val_loader = processor_c.create_loaders(X_train, X_val, y_train, y_val)

        # 2. 根据 task 变量执行特定任务
        if task == "train":
            pipeline.train(processor_c, train_loader, val_loader, epochs=30)

        elif task == "eval":
            pipeline.eval(val_loader, num_features=processor_c.num_features)

        elif task == "pred":
            # 拿验证集第一条原始数据作为演示
            mock_raw_sample = processor_c.scaler.inverse_transform(X_val[0].reshape(1, -1))[0]
            pred_class = pipeline.pred(mock_raw_sample, processor_c.scaler)
            print(f">> [动态预测成功] 原始数据物理标签预测结果为: Class {pred_class[0]}")

        else:
            print(f"未知任务类型: {task}，请选择 'train', 'eval' 或 'pred'")

    else:
        print(f"未在指定路径找到文件: {meter_c_path}")