import os
import random
from pathlib import Path
import yaml
from ultralytics import RTDETR  # 替换为 RTDETR


class RTDETRDefectDetector:
    """焊接缺陷检测器类 (基于 RT-DETR，复用你的动态子集机制)"""

    def __init__(self, dataset_root: str, model_weight: str = "rtdetr-l.pt"):
        self.dataset_root = Path(dataset_root)
        self.model_weight = model_weight
        self.yaml_path = self.dataset_root / "dataset.yaml"
        self.model = None

        self.id_to_en = {
            0: "Porosity", 1: "Inclusion", 2: "Under-cut", 3: "Burn-through",
            4: "Crack", 5: "Overlap", 6: "Reference 1", 7: "Reference 2",
            8: "Reference 3", 9: "Hidden porosity", 10: "Shrinkage cavity",
            11: "Lack of fusion", 12: "Incomplete root penetration"
        }

        self.image_dir = self.dataset_root / "train" / "images"
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.all_images = [str(p) for p in self.image_dir.iterdir() if p.suffix.lower() in valid_extensions]

        print(f"📂 原始数据集中共找到 {len(self.all_images)} 张图片。")
        self._load_model()

    def _prepare_subset_yaml(self, sample_count: int, txt_name: str):
        actual_count = min(sample_count, len(self.all_images))
        selected_images = random.sample(self.all_images, actual_count)

        # === 修改这里：如果是训练模式(传入的 sample_count 是 1000)，我们切分成 900 张训练和 100 张验证 ===
        if sample_count == 1000:
            train_list = selected_images[:900]
            val_list = selected_images[900:]

            # 写入两个独立的列表文件
            with open(self.dataset_root / "train_sub.txt", 'w', encoding='utf-8') as f:
                for img_path in train_list: f.write(img_path + '\n')
            with open(self.dataset_root / "val_sub.txt", 'w', encoding='utf-8') as f:
                for img_path in val_list: f.write(img_path + '\n')

            yaml_content = {
                'path': str(self.dataset_root),
                'train': "train_sub.txt",
                'val': "val_sub.txt",
                'names': self.id_to_en
            }
        else:
            # === 如果是普通的评估模式(只传了 50 张)，维持原样不变 ===
            txt_path = self.dataset_root / txt_name
            with open(txt_path, 'w', encoding='utf-8') as f:
                for img_path in selected_images: f.write(img_path + '\n')
            yaml_content = {
                'path': str(self.dataset_root),
                'train': txt_name,
                'val': txt_name,
                'names': self.id_to_en
            }

        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True, default_flow_style=False)

    def _load_model(self):
        print(f"⏳ 正在初始化 RT-DETR 模型 (权重: {self.model_weight}) ...")
        self.model = RTDETR(self.model_weight)
        if hasattr(self.model, 'model') and self.model.model is not None:
            self.model.model.names = self.id_to_en

    def train(self, epochs: int = 8, batch: int = 4, imgsz: int = 640, device: str = "0", sample_count: int = 1000):
        # 注意：RT-DETR 比 YOLO 稍重，如果显存不够（比如 8G 以下），把 batch 调小（如 2 或 4）
        print(f"\n🚀 [TRAIN] 正在准备 {sample_count} 张图片的训练子集...")
        self._prepare_subset_yaml(sample_count=sample_count, txt_name="train_sub.txt")

        results = self.model.train(
            data=str(self.yaml_path),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            workers=0,
            project="RWDD_RTDETR",
            name="train_rtdetr_l"
        )
        return results

    def evaluate(self, device: str = "0", sample_count: int = 50):
        print(f"\n📊 [EVALUATE] 正在准备 {sample_count} 张图片的评估子集...")
        self._prepare_subset_yaml(sample_count=sample_count, txt_name="val_sub.txt")

        metrics = self.model.val(
            data=str(self.yaml_path),
            device=device,
            workers=0,
            project="RWDD_RTDETR",
            name="val_rtdetr_l"
        )
        # 打印各类别查准率、查全率、F1-Score
        print("各类别查准率(P):", metrics.box.p)
        print("各类别查全率(R):", metrics.box.r)
        print("各类别 F1-Score:", metrics.box.f1)
        return metrics


if __name__ == "__main__":
    DATASET_ROOT = r"D:\DataSets\InsideMachine\Radiographsweldingdefectdetection"

    # 你可以先选 "train" 训练 20 轮，再选 "evaluate" 评估
    MODE = "evaluate"

    if MODE == "train":
        detector = RTDETRDefectDetector(dataset_root=DATASET_ROOT, model_weight="rtdetr-l.pt")
        detector.train(epochs=8, batch=4, imgsz=640, device="0", sample_count=1000)
    elif MODE == "evaluate":
        # 训练完后换成你自己的 best.pt 路径
        BEST_WEIGHT = r"runs\detect\RWDD_RTDETR\train_rtdetr_l\weights\best.pt"
        detector = RTDETRDefectDetector(dataset_root=DATASET_ROOT, model_weight=BEST_WEIGHT)
        detector.evaluate(device="0", sample_count=50)