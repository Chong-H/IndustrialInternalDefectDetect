import os
import random
from pathlib import Path
import yaml
from ultralytics import YOLO


class WeldingDefectDetector:
    """焊接缺陷检测器类 (基于 YOLOv11n-seg，支持子集轻量化训练)"""

    def __init__(self, dataset_root: str, model_weight: str = "yolo11n-seg.pt"):
        """
        :param dataset_root: 数据集根目录路径
        :param model_weight: 初始权重路径
        """
        self.dataset_root = Path(dataset_root)
        self.model_weight = model_weight
        self.yaml_path = self.dataset_root / "dataset.yaml"
        self.model = None

        # 统一规范的英文标签（0-12）
        self.id_to_en = {
            0: "Porosity", 1: "Inclusion", 2: "Under-cut", 3: "Burn-through",
            4: "Crack", 5: "Overlap", 6: "Reference 1", 7: "Reference 2",
            8: "Reference 3", 9: "Hidden porosity", 10: "Shrinkage cavity",
            11: "Lack of fusion", 12: "Incomplete root penetration"
        }

        # 获取所有可用的图片文件
        self.image_dir = self.dataset_root / "train" / "images"
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.all_images = [str(p) for p in self.image_dir.iterdir() if p.suffix.lower() in valid_extensions]

        print(f"📂 原始数据集 train/images 中共有 {len(self.all_images)} 张图片。")
        self._load_model()

    def _prepare_subset_yaml(self, sample_count: int, txt_name: str):
        """
        内部方法：随机抽取指定数量的图片，生成临时 txt 列表，并更新映射的 dataset.yaml
        """
        if not self.all_images:
            raise ValueError("❌ 错误：未在指定路径找到任何有效图片！")

        actual_count = min(sample_count, len(self.all_images))
        selected_images = random.sample(self.all_images, actual_count)

        # 1. 写入临时的图片路径列表文件 (例如 train_sub.txt 或 val_sub.txt)
        txt_path = self.dataset_root / txt_name
        with open(txt_path, 'w', encoding='utf-8') as f:
            for img_path in selected_images:
                f.write(img_path + '\n')

        print(f"📝 已随机抽样 {actual_count} 张图片写入路径列表: {txt_path}")

        # 2. 动态生成或更新给 YOLO 读的 yaml 配置文件
        # 让 train 和 val 直接指向这个含有具体图片路径的 txt 文件
        yaml_content = {
            'path': str(self.dataset_root),
            'train': txt_name,
            'val': txt_name,
            'names': self.id_to_en
        }
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True, default_flow_style=False)
        print(f"⚙️  YOLO 配置文件已动态更新: {self.yaml_path}")

    def _load_model(self):
        """内部方法：加载 YOLOv11n-seg 模型"""
        print(f"⏳ 正在初始化 YOLOv11n-seg 模型 (权重: {self.model_weight}) ...")
        self.model = YOLO(self.model_weight)
        if hasattr(self.model, 'model') and self.model.model is not None:
            self.model.model.names = self.id_to_en

    def train(self, epochs: int = 20, batch: int = 16, imgsz: int = 640, device: str = "0", sample_count: int = 1000):
        """
        抽取指定数量的图片进行训练
        """
        print(f"\n🚀 [TRAIN] 正在准备 1000 张图片的训练子集...")
        # 生成仅包含 1000 张图的训练配置
        self._prepare_subset_yaml(sample_count=sample_count, txt_name="train_sub.txt")

        print(f"🔥 开始训练模型 (Epochs={epochs}, Batch={batch}) ...")
        results = self.model.train(
            data=str(self.yaml_path),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            workers=0,
            project="RWDD_YOLOv11",
            name="train_yolo11n_seg"
        )
        print("🎉 训练完成！")
        return results

    def evaluate(self, device: str = "0", sample_count: int = 50):
        """
        抽取指定数量的图片进行评估验证
        """
        print(f"\n📊 [EVALUATE] 正在准备 50 张图片的评估子集...")
        # 生成仅包含 50 张图的评估配置
        self._prepare_subset_yaml(sample_count=sample_count, txt_name="val_sub.txt")

        print(f"🔍 开始在 50 张随机样本上评估模型性能...")
        metrics = self.model.val(
            data=str(self.yaml_path),
            device=device,
            workers=0,
            project="RWDD_YOLOv11",
            name="val_yolo11n_seg"
        )
        print("✅ 评估完成！")
        return metrics

    def predict_random_samples(self, sample_count: int = 50, conf: float = 0.25, device: str = "0"):
        """
        随机抽样预测图片并保存结果（带英文标签，防乱码组件）
        """
        print(f"\n🎲 [PREDICT] 随机抽取 {sample_count} 张图片进行可视化推理...")
        selected_images = random.sample(self.all_images, min(sample_count, len(self.all_images)))

        results = self.model.predict(
            source=selected_images,
            save=False,
            conf=conf,
            device=device
        )

        output_dir = Path("runs/segment/predict_yolo11n_fixed")
        output_dir.mkdir(parents=True, exist_ok=True)

        import cv2
        for result in results:
            result.names = self.id_to_en
            annotated_frame = result.plot()
            img_name = Path(result.path).name
            cv2.imwrite(str(output_dir / img_name), annotated_frame)

        print(f"✨ 预测图已成功导出至: {output_dir.resolve()}")


# ==========================================
# Main 控制流入口
# ==========================================
if __name__ == "__main__":
    # 1. 数据集基础路径
    DATASET_ROOT = r"D:\DataSets\InsideMachine\Radiographsweldingdefectdetection"

    # 2. 【模式选择】 "train" 训练 / "evaluate" 评估
    # MODE = "train"
    MODE = "evaluate"

    # 3. 运行对应模式
    if MODE == "train":
        MODEL_WEIGHT = "yolo11n-seg.pt"  # 训练用官方预训练权重
        detector = WeldingDefectDetector(dataset_root=DATASET_ROOT, model_weight=MODEL_WEIGHT)

        # 执行训练：设置 epoch=20，从数据集中随机抽 sample_count=1000 张图
        detector.train(epochs=20, batch=8, imgsz=640, device="0", sample_count=1000)

    elif MODE == "evaluate":
        # 评估时：指向您训练出来的最佳模型权重
        MODEL_WEIGHT = r"runs\segment\RWDD_YOLOv11\train_yolo11n_seg-2\weights\best.pt"

        if not os.path.exists(MODEL_WEIGHT):
            print(f"❌ 错误：未找到权重文件 [{MODEL_WEIGHT}]，请先运行 train 模式训练模型！")
        else:
            detector = WeldingDefectDetector(dataset_root=DATASET_ROOT, model_weight=MODEL_WEIGHT)
            # 执行评估：只选 sample_count=50 张图片进行验证
            detector.evaluate(device="0", sample_count=50)
            # 评估完，顺便可视化输出这 50 张英文标签图方便查看效果
            detector.predict_random_samples(sample_count=50, device="0")