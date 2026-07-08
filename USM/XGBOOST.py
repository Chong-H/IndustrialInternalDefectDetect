import os
import re
import glob
import joblib  # 用于传统机器学习模型和持久化组件的序列化
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 核心模型算法导入
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ==========================================
# 1. 动态数据加载与解析模块
# ==========================================
class MLUSMProcessor:
    """自适应数据处理器：自动识别特征维度与类别总数（专为树模型优化，无需Scaler）"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.num_features = 0
        self.num_classes = 0

    def load_and_adapt(self, test_size: float = 0.2, random_state: int = 42):
        df = pd.read_csv(self.data_path, header=0)
        df.dropna(how='all', inplace=True)

        # 物理切片：剔除第0列(id)和最后一列(标签)
        X = df.iloc[:, 1:-1].values.astype(float)
        y = df.iloc[:, -1].astype(int).values

        self.num_features = X.shape[1]
        self.num_classes = len(np.unique(y))

        print(
            f"📊 成功读取 CSV 数据！总样本数: {X.shape[0]} | 纯特征维度: {self.num_features} | 类别数: {self.num_classes}")

        # 将 1-based 标签转化为 0-based 索引以兼容 XGBoost
        y = y - 1

        # 树模型不需要进行 StandardScaler 标准化，直接切分返回
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ==========================================
# 2. 动态树模型路由与核心管道
# ==========================================
class MLUSMPipeline:
    """动态流水线：支持自动文件名命名持久化与基于维度的模型自动路由"""

    def __init__(self, model_type: str = "xgboost", model_dir: str = "./saved_ml_models"):
        self.model_type = model_type.lower().strip()
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        if self.model_type not in ["rf", "xgboost"]:
            raise ValueError("❌ model_type 必须是 'rf' (随机森林) 或 'xgboost' (XGBoost)")

    def _generate_model_name(self, num_features: int, num_classes: int) -> str:
        """动态生成唯一的标准化模型文件名"""
        prefix = "RF" if self.model_type == "rf" else "XGBoost"
        return os.path.join(self.model_dir, f"USM_{prefix}_Features{num_features}_Classes{num_classes}.pkl")

    def _auto_route_and_load_model(self, num_features: int) -> object:
        """核心路由机制：在磁盘中扫描符合当前特征输入维度的 .pkl 文件并自动加载"""
        prefix = "RF" if self.model_type == "rf" else "XGBoost"
        search_pattern = os.path.join(self.model_dir, f"USM_{prefix}_Features{num_features}_Classes*.pkl")
        matched_files = glob.glob(search_pattern)

        if not matched_files:
            raise FileNotFoundError(
                f"❌ 错误: 未能在目录 {self.model_dir} 中找到符合特征维度 {num_features} 的 {prefix} 模型。")

        target_model_path = matched_files[0]
        print(f"--> [智能路由] 成功匹配权重文件: {os.path.basename(target_model_path)}")

        # 加载持久化模型
        return joblib.load(target_model_path)

    def train(self, processor: MLUSMProcessor, X_train, y_train, X_val, y_val):
        """完全解耦的训练接口，动态初始化模型并导出带维度的权重文件"""
        num_features = processor.num_features
        num_classes = processor.num_classes
        save_path = self._generate_model_name(num_features, num_classes)

        # 动态初始化具体的分类器
        if self.model_type == "rf":
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            print(f"\n================ 启动自适应 [随机森林] 训练 ================")
        else:
            # XGBoost 多分类需要特别指定 eval_metric
            model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                  eval_metric='mlogloss', random_state=42, n_jobs=-1)
            print(f"\n================ 启动自适应 [XGBoost] 训练 ================")

        print(f"数据指纹 -> 特征列数: {num_features} | 故障类别数: {num_classes}")

        # 树模型一步训练到位
        model.fit(X_train, y_train)

        # 固化并保存模型
        joblib.dump(model, save_path)
        print(f"⭐ 模型已成功序列化并保存至: {save_path}")

    def eval(self, X_val, y_val, num_features: int):
        """评估接口：无需指定模型实例，输入验证数据流和特征维度，自动路由匹配"""
        model = self._auto_route_and_load_model(num_features)
        all_preds = model.predict(X_val)

        target_names = [f"Class {i + 1}" for i in sorted(np.unique(y_val))]

        print(f"\n==================== 📊 详细评估报告 [{self.model_type.upper()}] ====================")
        print(classification_report(y_val, all_preds, target_names=target_names, digits=4))

        print("==================== 🧱 混淆矩阵 (Confusion Matrix) ====================")
        print(confusion_matrix(y_val, all_preds))
        print("========================================================")

    def pred(self, raw_features: np.ndarray) -> np.ndarray:
        """预测接口：根据输入的原始特征维度自动加载对应权重，无缝支持单样本或批量多样本推理"""
        if raw_features.ndim == 1:
            raw_features = raw_features.reshape(1, -1)

        num_features = raw_features.shape[1]
        model = self._auto_route_and_load_model(num_features)

        preds = model.predict(raw_features)
        # 加上 1 还原回物理故障标签范围 1, 2, 3, 4
        return preds + 1


# ==========================================
# 3. 自动化流水线执行验证入口
# ==========================================
if __name__ == "__main__":
    # 基础配置
    DATA_DIR = r"D:\DataSets\InsideMachine\UltrasonicFlowmeterDiagnosticsDataSet"
    meter_c_path = os.path.join(DATA_DIR, "flowmeter_B_mined.csv")

    # ----------------------------------------------------
    # 控制变量区域：在这里修改任务和模型架构
    # ----------------------------------------------------
    task = "eval"  # 可选: "train" | "eval" | "pred"
    # task = "train"  # 可选: "train" | "eval" | "pred"
    model_type = "rf"  # 可选: "xgboost" | "rf" (随机森林)

    # 初始化控制流水线
    pipeline = MLUSMPipeline(model_type=model_type, model_dir="./saved_ml_models")

    # 1. 动态加载解析
    if os.path.exists(meter_c_path):
        processor_c = MLUSMProcessor(data_path=meter_c_path)
        X_train, X_val, y_train, y_val = processor_c.load_and_adapt(test_size=0.2)

        # 2. 根据 task 变量执行特定任务
        if task == "train":
            pipeline.train(processor_c, X_train, y_train, X_val, y_val)

        elif task == "eval":
            pipeline.eval(X_val, y_val, num_features=processor_c.num_features)

        elif task == "pred":
            # 拿验证集第一条原始数据作为演示（由于没建立 Scaler，直接取原始矩阵数据）
            mock_raw_sample = X_val[0]
            pred_class = pipeline.pred(mock_raw_sample)
            print(f">> [动态预测成功] 原始数据物理标签预测结果为: Class {pred_class[0]}")

        else:
            print(f"未知任务类型: {task}，请选择 'train', 'eval' 或 'pred'")
    else:
        print(f"未在指定路径找到文件: {meter_c_path}")