import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.models.segmentation as seg_models


class WeldMaskDataset(Dataset):
    """标准语义分割数据集加载器：专吃原始图片与 PNG 掩码图"""

    def __init__(self, img_dir, mask_dir, imgsz=520):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgsz = imgsz
        # 1. 过滤获取所有的图片文件名
        all_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # 2. 核心修改：使用切片强行只取前 200 张，多余的直接抛弃
        self.filenames = all_files[:200]

        print(f"[DATA] 文件夹内共有 {len(all_files)} 张图，已过滤并强制限制加载前 {len(self.filenames)} 张进行快速实验。")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 读取图像并归一化到 0~1
        image = cv2.imread(img_path)
        image = cv2.resize(image, (self.imgsz, self.imgsz))
        image = image.transpose(2, 0, 1) / 255.0  # HWC -> CHW

        # 读取最原始的黑白掩码图
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.int64)
        else:
            mask = np.zeros((self.imgsz, self.imgsz), dtype=np.int64)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long), img_name


class FCNWeldPipeline:
    """基于 torchvision 官方内置库的经典 FCN 分割管线"""

    def __init__(self, weights_path: str = "fcn_weld_best.pth", device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path

        print("[INFO] 正在从 torchvision 官方库加载经典 FCN 架构...")
        # 调用官方 FCN-ResNet50 架构并禁用水印下载
        self.model = seg_models.fcn_resnet50(num_classes=2, weights_backbone=None).to(self.device)

    def train(self, img_dir, mask_dir, epochs=20, batch_size=4, lr=1e-4):
        print("\n" + "="*50)
        print(f"[INFO] 正在启动 FCN 缺陷分割训练...")
        print(f" -> 设定总训练轮数 (Epochs): {epochs}")
        print(f" -> 每批次大小 (Batch Size):  {batch_size}")
        print("="*50)

        dataset = WeldMaskDataset(img_dir, mask_dir, imgsz=520)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        best_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            print(f" ▶▶▶ 正在执行：第 [{epoch+1}/{epochs}] 轮训练...")

            for batch_idx, (images, masks, _) in enumerate(dataloader):
                images, masks = images.to(self.device), masks.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)['out']
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # 🌟 核心改进：原生打印进度，不再卡着不动
                if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
                    percent = int((batch_idx + 1) / len(dataloader) * 100)
                    print(f"   [FCN 轮次进度]: {percent:3d}% | 批次: [{batch_idx+1}/{len(dataloader)}] | 当前批 Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / len(dataloader)
            print(f" 🏁 [FCN 轮次总结] 第 {epoch+1} 轮完成 -> 平均 Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), self.weights_path)
                print(f"   [SAVE] 最优 FCN 权重已更新保存至: {self.weights_path}")
            print("-" * 50)

        print(f"\n[SUCCESS] FCN 训练任务全部安全结束！")

    def evaluate(self, img_dir, mask_dir):
        if os.path.exists(self.weights_path):
            print(f"[INFO] 正在载入最优权重进行 FCN 指标计算: {self.weights_path}")
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        else:
            print("[WARNING] 未找到权重文件，将以未训练的初始状态进行指标评估。")

        dataset = WeldMaskDataset(img_dir, mask_dir, imgsz=520)
        indices = list(range(min(50, len(dataset))))
        sub_dataset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(sub_dataset, batch_size=1, shuffle=False)

        self.model.eval()
        tp, fp, fn, tn = 0, 0, 0, 0

        print("\n" + "=" * 50 + "\n 开始计算 50 张图的 FCN 精密像素级指标... \n" + "=" * 50)
        with torch.no_grad():
            for images, masks, _ in dataloader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)['out']
                preds = outputs.argmax(dim=1)

                tp += ((preds == 1) & (masks == 1)).sum().item()
                fp += ((preds == 1) & (masks == 0)).sum().item()
                fn += ((preds == 0) & (masks == 1)).sum().item()
                tn += ((preds == 0) & (masks == 0)).sum().item()

        # 🌟 核心改进：引入 1e-7，防止模型还没练到位、输出都为 0 时引发除以零崩溃
        eps = 1e-7
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)

        print("\n" + "=" * 50)
        print(" 官方 FCN 焊接缺陷分割性能评估报告 (50张样本) ")
        print("=" * 50)
        print(f" -> 像素查准率 (Precision): {precision:.4f}")
        print(f" -> 像素查全率 (Recall):    {recall:.4f}")
        print(f" -> F1-Score:              {f1:.4f}")
        print(f" -> 缺陷交并比 (IoU / mAP): {iou:.4f}")
        print("=" * 50)

    def predict(self, source_image_path: str, save_dir: str = "fcn_predictions"):
        """对单张输入图片进行推理，并渲染叠加缺陷边缘输出保存"""
        if os.path.exists(self.weights_path):
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.eval()

        os.makedirs(save_dir, exist_ok=True)

        ori_img = cv2.imread(source_image_path)
        h, w, _ = ori_img.shape

        img = cv2.resize(ori_img, (520, 520))
        img = img.transpose(2, 0, 1) / 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)['out']
            preds = outputs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        pred_mask = cv2.resize(preds, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_color = np.zeros_like(ori_img)
        mask_color[pred_mask == 1] = [0, 0, 255]  # FCN 预测出的缺陷用红色高亮方便做实验对比
        overlay = cv2.addWeighted(ori_img, 0.7, mask_color, 0.3, 0)

        save_path = os.path.join(save_dir, os.path.basename(source_image_path))
        cv2.imwrite(save_path, overlay)
        print(f"[INFO] 预测渲染完成！结果已保存至: {save_path}")


if __name__ == "__main__":
    # 想要训练时改成 "train"，想要评估时改成 "evaluate"，想要画图看改 "predict"
    task = "evaluate"

    img_dir = r"D:\DataSets\InsideMachine\Weld-defect-detection-datasets\datasets1\datasets1\images"
    mask_dir = r"D:\DataSets\InsideMachine\Weld-defect-detection-datasets\datasets1\datasets1\labels_mask"

    pipeline = FCNWeldPipeline(weights_path="fcn_weld_best.pth", device="cuda:0")

    if task == "train":
        pipeline.train(img_dir=img_dir, mask_dir=mask_dir, epochs=5, batch_size=4) # 设定和之前一样的 5 轮
    elif task == "evaluate":
        pipeline.evaluate(img_dir=img_dir, mask_dir=mask_dir)
    elif task == "predict":
        test_img = os.path.join(img_dir, os.listdir(img_dir)[0])
        pipeline.predict(source_image_path=test_img)