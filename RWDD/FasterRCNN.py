import os
import time
from pathlib import Path
import ssl

import random as py_random

# 禁用SSL证书校验，解决下载权重SSL报错
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
import cv2
import numpy as np


# ==============================================================================
# 1. 数据集加载适配层 (保持原生)
# ==============================================================================
class FasterRCNNDataset(Dataset):
    def __init__(self, txt_list_path):
        with open(txt_list_path, 'r', encoding='utf-8') as f:
            self.img_paths = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"❌ 无法读取图片: {img_path}")

        h, w, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt").replace(
            ".jpeg", ".txt")
        boxes, labels = [], []

        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue

                    try:
                        cls_id = int(parts[0])
                        # 严格限制 12 种缺陷范围 (0 到 12)
                        if cls_id < 0 or cls_id > 12:
                            continue

                        # 兼容多边形或标定不规范数据
                        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                        if len(coords) < 4:
                            continue

                        xs = coords[0::2] * w
                        ys = coords[1::2] * h

                        xmin = float(np.min(xs))
                        ymin = float(np.min(ys))
                        xmax = float(np.max(xs))
                        ymax = float(np.max(ys))

                        # 强行纠正颠倒的坐标轴顺序
                        if xmin > xmax: xmin, xmax = xmax, xmin
                        if ymin > ymax: ymin, ymax = ymax, ymin

                        # 边缘严格裁剪防越界
                        xmin, ymin = max(0.0, xmin), max(0.0, ymin)
                        xmax, ymax = min(float(w), xmax), min(float(h), ymax)

                        # 🚨【最关键改动】：卡死边界框的最小长宽！必须大于 2 个绝对像素
                        # 如果宽高小于 2 像素，说明是无效噪声框/退化框，Faster R-CNN 必炸，必须跳过！
                        if (xmax - xmin) >= 2.0 and (ymax - ymin) >= 2.0:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(cls_id + 1)  # 0变为1，12变为13。背景保留0
                    except Exception:
                        continue
        if len(boxes) == 0:
            target = {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64)}
        else:
            target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                      "labels": torch.tensor(labels, dtype=torch.int64)}
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ==============================================================================
# 2. 面向对象重构：FasterRCNNDefectDetector 类
# ==============================================================================
class FasterRCNNDefectDetector:
    """焊接缺陷检测器类 (基于 Faster R-CNN，完美封装 OOP 架构)"""

    def __init__(self, dataset_root: str, model_weight: str = "default"):
        self.dataset_root = Path(dataset_root)
        self.model_weight = model_weight
        self.train_txt_path = self.dataset_root / "train_sub.txt"
        self.val_txt_path = self.dataset_root / "val_sub.txt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        self.id_to_en = [
            "Porosity", "Inclusion", "Under-cut", "Burn-through", "Crack", "Overlap",
            "Reference 1", "Reference 2", "Reference 3", "Hidden porosity",
            "Shrinkage cavity", "Lack of fusion", "Incomplete root penetration"
        ]

        # 🚨 加上这三行：让 Faster R-CNN 自己也能看到源数据
        self.image_dir = self.dataset_root / "train" / "images"
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.all_images = [str(p) for p in self.image_dir.iterdir() if p.suffix.lower() in valid_extensions]

        self._load_model()

    def _prepare_evaluate_subset(self, sample_count: int = 300):
        """专门为评估抽样指定数量（300张）的独立图片"""
        print(f"⚙️ Faster R-CNN 正在重新抽样 {sample_count} 张图片用于大规模评估盘点...")

        # 1. 优先排除掉参与过训练的 900 张图，防止信息泄露
        trained_images = set()
        if self.train_txt_path.exists():
            with open(self.train_txt_path, 'r', encoding='utf-8') as f:
                trained_images = set([line.strip() for line in f.readlines() if line.strip()])

        remain_images = [img for img in self.all_images if img not in trained_images]

        # 如果剩余未训练的图不够 300 张，则从全部图片里抽
        source_pool = remain_images if len(remain_images) >= sample_count else self.all_images

        # 2. 随机抽取 300 张
        selected_images = py_random.sample(source_pool, min(sample_count, len(source_pool)))

        # 3. 强行覆写进验证文本中
        with open(self.val_txt_path, 'w', encoding='utf-8') as f:
            for img_path in selected_images:
                f.write(img_path + '\n')
        print(f"✅ 评估集重新锁定：已随机抽取 {len(selected_images)} 张图写入 val_sub.txt。")
    def _prepare_subset_data(self, sample_count: int = 1000):
        """严格限制总数为 1000 张图：900张用于训练，100张用于验证"""
        print(f"⚙️ Faster R-CNN 正在自主清洗并重新生成 1000 张子集数据...")
        selected_images = py_random.sample(self.all_images, min(sample_count, len(self.all_images)))

        train_list = selected_images[:900]
        val_list = selected_images[900:]

        with open(self.train_txt_path, 'w', encoding='utf-8') as f:
            for img_path in train_list: f.write(img_path + '\n')
        with open(self.val_txt_path, 'w', encoding='utf-8') as f:
            for img_path in val_list: f.write(img_path + '\n')
        print(f"✅ 数据重新锁定：900张写入 train_sub.txt, 100张写入 val_sub.txt。")
    def _load_model(self):
        """内部方法：加载与构建 Faster R-CNN 模型"""
        print(f"⏳ 正在初始化 Faster R-CNN 架构 ...")
        # 12 类缺陷 + 1 类背景 = 13 类
        num_classes = 14

        if self.model_weight == "default":
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            print("💡 已成功挂载官方预训练骨干网络权重。")
        else:
            # 离线加载本地训练好的权重
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes)
            self.model.load_state_dict(torch.load(self.model_weight, map_location=self.device))
            print(f"💾 已成功从本地加载固化权重: {self.model_weight}")

        self.model.to(self.device)

    def train(self, epochs: int = 8, batch: int = 4, device: str = "0"):

        self._prepare_subset_data(sample_count=1000)
        """
        根据之前抽样生成的 train_sub.txt 执行模型对比训练
        """
        print(f"\n🚀 [TRAIN] 开始 Faster R-CNN 对比训练 (严格限制: {epochs}个 Epoch, batch={batch}) ...")

        if not self.train_txt_path.exists():
            raise FileNotFoundError(f"❌ 找不到抽样好的训练集文件 [{self.train_txt_path}]，请确认之前已运行过抽样数据！")

        train_dataset = FasterRCNNDataset(self.train_txt_path)
        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=0)

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            start_time = time.time()

            for images, targets in train_loader:
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()

            print(
                f"Epoch [{epoch + 1}/{epochs}] - 耗时: {time.time() - start_time:.1f}s - 平均 Loss: {epoch_loss / len(train_loader):.4f}")

        # 固化权重存盘
        output_dir = Path("runs/faster_rcnn")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "faster_rcnn_best.pt"
        torch.save(self.model.state_dict(), save_path)
        print(f"🎉 训练完成！新权重已安全固化至: {save_path}")

    @torch.no_grad()
    def evaluate(self, iou_threshold: float = 0.5, conf_threshold: float = 0.25):
        self._prepare_evaluate_subset(sample_count=300)

        print(f"\n📊 [EVALUATE] 开始在独立的 300 张验证图上盘点 Faster R-CNN 性能指标...")

        # 🚨 确保这里的 batch_size 可以设为 6 或 8（跑 300 张图会比 batch=4 快一倍）
        val_dataset = FasterRCNNDataset(self.val_txt_path)
        val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, collate_fn=collate_fn, num_workers=0)
        """
        根据之前抽样生成的 val_sub.txt 对 100 张验证集进行精准的 P, R, F1 指标统计结算
        """
        print(f"\n📊 [EVALUATE] 开始在独立的 100 张验证图上盘点 Faster R-CNN 性能指标...")

        if not self.val_txt_path.exists():
            raise FileNotFoundError(f"❌ 找不到抽样好的验证集文件 [{self.val_txt_path}]，请先运行数据准备流！")

        val_dataset = FasterRCNNDataset(self.val_txt_path)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

        self.model.eval()
        num_classes = 13
        tp = torch.zeros(num_classes)
        fp = torch.zeros(num_classes)
        fn = torch.zeros(num_classes)

        for images, targets in val_loader:
            images = list(img.to(self.device) for img in images)
            outputs = self.model(images)

            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].to(self.device)
                gt_labels = target["labels"].to(self.device) - 1  # 剥离背景层，还原至 0-11

                pred_boxes = output["boxes"]
                pred_labels = output["labels"] - 1
                pred_scores = output["scores"]

                keep = pred_scores >= conf_threshold
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                for c in range(num_classes):
                    c_gt_boxes = gt_boxes[gt_labels == c]
                    c_pred_boxes = pred_boxes[pred_labels == c]

                    num_gt = c_gt_boxes.shape[0]
                    num_pred = c_pred_boxes.shape[0]

                    if num_gt == 0:
                        fp[c] += num_pred
                        continue
                    if num_pred == 0:
                        fn[c] += num_gt
                        continue

                    iou_matrix = box_iou(c_pred_boxes, c_gt_boxes)
                    matched_gt = set()

                    for p_idx in range(num_pred):
                        max_iou, max_g_idx = iou_matrix[p_idx].max(dim=0)
                        if max_iou >= iou_threshold and max_g_idx.item() not in matched_gt:
                            tp[c] += 1
                            matched_gt.add(max_g_idx.item())
                        else:
                            fp[c] += 1

                    fn[c] += (num_gt - len(matched_gt))

        p = (tp / (tp + fp + 1e-6)).cpu().numpy()
        r = (tp / (tp + fn + 1e-6)).cpu().numpy()
        f1 = (2 * p * r / (p + r + 1e-6))

        # 漂亮化打印
        print("\n" + "=" * 60)
        print("🎯 Faster R-CNN 最终评估结果 (IoU=0.5, Conf=0.25):")
        print("-" * 60)
        print(f"{'缺陷英文名称':<28}{'查准率(P)':<12}{'查全率(R)':<12}{'F1-Score':<12}")
        print("-" * 60)
        for i in range(num_classes):
            print(f"{self.id_to_en[i]:<28}{p[i]:<12.4f}{r[i]:<12.4f}{f1[i]:<12.4f}")
        print("-" * 60)
        print(f"{'全类别平均(Macro Average)':<28}{p.mean():<12.4f}{r.mean():<12.4f}{f1.mean():<12.4f}")
        print("=" * 60)


# ==========================================
# 外部简洁的 OOP 调用流入口
# ==========================================
if __name__ == "__main__":
    DATASET_ROOT = r"D:\DataSets\InsideMachine\Radiographsweldingdefectdetection"

    # 模式选择
    # MODE = "train"
    MODE = "evaluate"

    if MODE == "train":
        # 实例化类（采用默认预训练骨干网络）
        detector = FasterRCNNDefectDetector(dataset_root=DATASET_ROOT, model_weight="default")
        # 外部调用一键训练
        detector.train(epochs=8, batch=4, device="0")
        # 训练结束后顺带输出这 100 张验证图的 P, R, F1 报表
        detector.evaluate()

    elif MODE == "evaluate":
        # 评估阶段，指定本地的最佳固化权重路径
        BEST_WEIGHT = r"runs\faster_rcnn\faster_rcnn_best.pt"
        if not os.path.exists(BEST_WEIGHT):
            print(f"❌ 错误：未找到本地权重文件 [{BEST_WEIGHT}]，请先切换为 train 模式跑完对比实验！")
        else:
            detector = FasterRCNNDefectDetector(dataset_root=DATASET_ROOT, model_weight=BEST_WEIGHT)
            detector.evaluate()