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

        # 2. 🌟 核心修改：使用切片强行只取前 200 张，多余的直接抛弃
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
            mask = (mask > 127).astype(np.int64)  # 0代表背景，1代表缺陷
        else:
            mask = np.zeros((self.imgsz, self.imgsz), dtype=np.int64)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long), img_name


class DeepLabWeldPipeline:
    """基于 torchvision 官方内置库的工业级 DeepLabV3 分割管线"""

    def __init__(self, weights_path: str = "deeplabv3_weld_best.pth", device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path

        print("[INFO] 正在从 torchvision 本地 官方库加载标准 DeepLabV3 架构...")
        # 使用 ResNet50 作为 Backbone，二分类输出
        self.model = seg_models.deeplabv3_resnet50(num_classes=2, weights_backbone=None).to(self.device)

    def train(self, img_dir, mask_dir, epochs=20, batch_size=4, lr=1e-4):
        # 1. 启动时立即打印核心配置看板
        print("\n" + "=" * 50)
        print(f"[INFO] 正在启动语义分割训练引擎...")
        print(f" -> 当前算法网络: {self.__class__.__name__}")
        print(f" -> 设定总训练轮数 (Epochs): {epochs}")
        print(f" -> 每批次大小 (Batch Size):  {batch_size}")
        print(f" -> 学习率 (Learning Rate):  {lr}")
        print(f" -> 计算核心设备 (Device):    {self.device}")
        print("=" * 50)

        dataset = WeldMaskDataset(img_dir, mask_dir, imgsz=520)
        # 强制单线程加载 num_workers=0，彻底规避 Windows 死锁挂起问题
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

        print(f"[INFO] 焊接数据集加载成功！样本总计: {len(dataset)} 张 | 每轮训练总批数: {len(dataloader)} steps\n")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        best_loss = float('inf')

        # 2. 进入主体训练循环
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            print(f" ▶▶▶ 正在执行：第 [{epoch + 1}/{epochs}] 轮训练...")

            # 纯原生循环，通过 batch_idx 计数来手动在命令行输出进度
            for batch_idx, (images, masks, _) in enumerate(dataloader):
                images, masks = images.to(self.device), masks.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)['out']
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # 核心：每完成 5 个 batch，在命令行强制刷新当前轮次的内部进度和单批 Loss
                if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
                    percent = int((batch_idx + 1) / len(dataloader) * 100)
                    print(
                        f"   [轮次进度]: {percent:3d}% | 批次: [{batch_idx + 1}/{len(dataloader)}] | 当前批 Loss: {loss.item():.4f}")

            # 3. 每轮结束，输出完整的总结报告
            avg_loss = epoch_loss / len(dataloader)
            print(f" 🏁 [轮次总结] 第 {epoch + 1} 轮跑完 -> 全局平均 CrossEntropy Loss: {avg_loss:.4f}")

            # 检查并自动保存最优模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), self.weights_path)
                print(f"   [SAVE] 🎯 监测到更低的 Loss 表现！最优权重已更新保存至: {self.weights_path}")
            print("-" * 50)

        print(f"\n[SUCCESS] 焊接缺陷模型训练任务全部安全结束！共计完成 {epochs} 轮。")

    def evaluate(self, img_dir, mask_dir):
        if os.path.exists(self.weights_path):
            print(f"[INFO] 正在载入最优权重进行 50 张图指标计算: {self.weights_path}")
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        else:
            print("[WARNING] 未找到权重文件，将以未训练的初始状态进行指标评估展示。")

        dataset = WeldMaskDataset(img_dir, mask_dir, imgsz=520)
        indices = list(range(min(50, len(dataset))))
        sub_dataset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(sub_dataset, batch_size=1, shuffle=False)

        self.model.eval()
        tp, fp, fn, tn = 0, 0, 0, 0

        print("\n" + "=" * 50 + "\n 开始计算 50 张图的 DeepLabV3 精密像素级指标... \n" + "=" * 50)
        with torch.no_grad():
            for images, masks, _ in dataloader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)['out']
                preds = outputs.argmax(dim=1)

                tp += ((preds == 1) & (masks == 1)).sum().item()
                fp += ((preds == 1) & (masks == 0)).sum().item()
                fn += ((preds == 0) & (masks == 1)).sum().item()
                tn += ((preds == 0) & (masks == 0)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        print("\n" + "=" * 50)
        print(" 官方 DeepLabV3 焊接缺陷分割性能评估报告 (50张样本) ")
        print("=" * 50)
        print(f" -> 像素查准率 (Precision): {precision:.4f}")
        print(f" -> 像素查全率 (Recall):    {recall:.4f}")
        print(f" -> F1-Score:              {f1:.4f}")
        print(f" -> 缺陷交并比 (IoU / mAP): {iou:.4f}")
        print("=" * 50)

    def predict(self, source_image_path: str, save_dir: str = "deeplabv3_predictions"):
        """对单张输入图片进行推理，并渲染叠加缺陷边缘输出保存"""
        if os.path.exists(self.weights_path):
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.eval()

        os.makedirs(save_dir, exist_ok=True)

        # 读取原图
        ori_img = cv2.imread(source_image_path)
        h, w, _ = ori_img.shape

        # 预处理
        img = cv2.resize(ori_img, (520, 520))
        img = img.transpose(2, 0, 1) / 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)['out']
            preds = outputs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # 将分割图尺寸拉伸回原图大小
        pred_mask = cv2.resize(preds, (w, h), interpolation=cv2.INTER_NEAREST)

        # 生成绿色半透明的缺陷图层叠加到原图上
        mask_color = np.zeros_like(ori_img)
        mask_color[pred_mask == 1] = [0, 255, 0]  # 缺陷用绿色高亮
        overlay = cv2.addWeighted(ori_img, 0.7, mask_color, 0.3, 0)

        save_path = os.path.join(save_dir, os.path.basename(source_image_path))
        cv2.imwrite(save_path, overlay)
        print(f"[INFO] 预测渲染完成！结果已保存至: {save_path}")


if __name__ == "__main__":
    # 【任务控制开关】: 可选 "train" (训练) 或 "evaluate" (评估指标) 或 "predict" (单张推理可视化)
    # task = "train"
    task = "evaluate"

    img_dir = r"D:\DataSets\InsideMachine\Weld-defect-detection-datasets\datasets1\datasets1\images"
    mask_dir = r"D:\DataSets\InsideMachine\Weld-defect-detection-datasets\datasets1\datasets1\labels_mask"

    pipeline = DeepLabWeldPipeline(weights_path="deeplabv3_weld_best.pth", device="cuda:0")

    if task == "train":
        pipeline.train(img_dir=img_dir, mask_dir=mask_dir, epochs=5, batch_size=4)
    elif task == "evaluate":
        pipeline.evaluate(img_dir=img_dir, mask_dir=mask_dir)
    elif task == "predict":
        # 推理单张图片示例（请替换为您目录里的真实图片名称）
        test_img = os.path.join(img_dir, os.listdir(img_dir)[0])
        pipeline.predict(source_image_path=test_img)