import os
from pathlib import Path
from ultralytics import YOLO
import shutil


class WeldDetector:
    """焊接缺陷检测器：包含模型训练与推理功能"""

    def __init__(self, model_path: str = "yolo11n-seg.pt", device: str = "0"):
        """初始化检测器

        :param model_path: 模型权重路径。训练时传预训练权重（如 yolo11n-seg.pt），
                           推理时传训练好的权重（如 best.pt）
        :param device: 运行设备，"0" 代表 GPU, "cpu" 代表 CPU
        """
        self.device = device
        self.model_path = model_path
        # 延迟加载模型，避免初始化时占用不必要的显存
        self.model = None

    def _lazy_load_model(self):
        """内部私有方法：确保模型已被加载"""
        if self.model == None:
            print(f"[INFO] 正在加载模型权重: {self.model_path}")
            self.model = YOLO(self.model_path)

    def train(self, yaml_path: str, epochs: int = 50, batch: int = 16, imgsz: int = 640):
        """执行模型训练"""
        self._lazy_load_model()
        print(f"[INFO] 开始训练，数据集配置: {yaml_path}")

        results = self.model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            workers=2,
            device=self.device,
            save=True,
            project="weld_outputs",
            name="yolo11n_seg_run",
            pretrained=True,
        )
        print("[INFO] 训练完成！")
        # 训练完后，自动将内部模型路径更新为刚生成的最佳权重
        best_path = Path("weld_outputs/yolo11n_seg_run/weights/best.pt")
        if best_path.exists():
            self.model_path = str(best_path)
            self.model = None  # 清空以便下次使用时重新加载新权重
            print(f"[INFO] 最佳权重已自动锁定: {best_path}")

    def predict(self, source: str, conf: float = 0.25, save: bool = True):
        """执行缺陷推理预测

        :param source: 预测源，可以是单张图片路径、文件夹路径、或者视频
        :param conf: 置信度阈值，低于该值的缺陷会被过滤掉
        :param save: 是否将渲染了渲染框和掩码的结果图片保存到本地
        :return: Ultralytics 的 Results 列表
        """
        self._lazy_load_model()
        print(f"[INFO] 正在对目标进行缺陷检测: {source}")

        # 执行推理
        results = self.model.predict(
            source=source,
            conf=conf,
            device=self.device,
            save=save,
            project="weld_predictions",
            name="predict_run",
            exist_ok=True,  # 如果文件夹存在，不新建而是覆盖/追加
        )

        # 简单的结果解析示例
        for r in results:
            boxes_count = len(r.boxes)
            masks_count = len(r.masks) if r.masks is not None else 0
            print(
                f" -> 文件: {os.path.basename(r.path)} | 检测到缺陷目标数: {boxes_count} | 成功生成掩码数: {masks_count}"
            )

        return results


# ==========================================
# 示例用法
# ==========================================
if __name__ == "__main__":
    # 执行任务控制：可选 "train" (训练模式) 或 "val" (验证模式)
    task = "val"
    #task = "train"

    # ----------------- 基础路径配置 -----------------
    # 预训练权重（用于训练）或 训练好的最佳权重（用于验证）
    pretrained_weight = "yolo11n-seg.pt"
    best_weight = r"C:\WorkSpace\py\IndustrialMachine\runs\segment\weld_outputs\yolo11n_seg_run\weights\best.pt"

    # 训练集配置文件路径（仅 train 模式使用）
    dataset_yaml = r"weld.yaml"

    # 原始数据集路径（仅 val 模式使用）
    img_dir = r"D:\DataSets\InsideMachine\Weld-defect-detection-datasets\datasets1\datasets1\images"
    txt_label_dir = r"D:\DataSets\InsideMachine\Weld-defect-detection-datasets\datasets1\datasets1\labels"

    # 动态评估的临时文件夹路径（仅 val 模式使用）
    eval_base = r"D:\DataSets\InsideMachine\Weld-defect-detection-datasets\datasets1\eval_temp"
    eval_img_dir = os.path.join(eval_base, "images")
    eval_lbl_dir = os.path.join(eval_base, "labels")

    # ==========================================
    # 流程 A：训练模式 (train)
    # ==========================================
    if task == "train":
        print("[INFO] 当前运行模式: 模型训练")
        detector = WeldDetector(model_path=pretrained_weight, device="0")
        detector.train(yaml_path=dataset_yaml, epochs=50)

    # ==========================================
    # 流程 B：验证模式 (val)
    # ==========================================
    elif task == "val":
        print("[INFO] 当前运行模式: 指标评估 (50张样本)")

        # ----------------- 1. 提取50个样本构建临时验证集 -----------------
        if not os.path.exists(best_weight):
            print(f"[ERROR] 未找到训练好的权重文件: {best_weight}，请检查路径。")
            exit()

        print("[INFO] 正在抽取50张图片及对应的YOLO格式标签进行指标比对...")
        os.makedirs(eval_img_dir, exist_ok=True)
        os.makedirs(eval_lbl_dir, exist_ok=True)

        # 获取所有图片列表并限制前50张
        all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        eval_images = all_images[:50]

        copied_count = 0
        for img_name in eval_images:
            base_name = os.path.splitext(img_name)[0]
            txt_name = base_name + ".txt"

            src_img = os.path.join(img_dir, img_name)
            src_txt = os.path.join(txt_label_dir, txt_name)

            # 确保图片和对应的txt标签同时存在才复制
            if os.path.exists(src_img) and os.path.exists(src_txt):
                shutil.copy(src_img, os.path.join(eval_img_dir, img_name))
                shutil.copy(src_txt, os.path.join(eval_lbl_dir, txt_name))
                copied_count += 1

        print(f"[INFO] 成功准备了 {copied_count} 组图片与标签用于评估。")

        if copied_count == 0:
            print("[ERROR] 抽取失败。请确认 labels 路径下存放的是转换后的 .txt 文件，而非 .png 掩码图片！")
            exit()

        # ----------------- 2. 动态生成评估所需的 yaml 配置文件 -----------------
        eval_yaml_path = os.path.join(eval_base, "eval_config.yaml")
        with open(eval_yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: {eval_base}\n")
            f.write(f"train: images\n")
            f.write(f"val: images\n")  # 验证集指向刚才创建的这50张图
            f.write(f"names:\n  0: weld_defect\n")

        # ----------------- 3. 加载模型并利用 YOLO 官方 Val 机制计算精密指标 -----------------
        detector = WeldDetector(model_path=best_weight, device="0")
        detector._lazy_load_model()  # 载入模型

        print("\n" + "=" * 50)
        print(" 开始计算 50 张图的查准率、查全率、F1和mAP50... ")
        print("=" * 50)

        # 调用官方的评估函数
        metrics = detector.model.val(
            data=eval_yaml_path,
            imgsz=640,
            device="0",
            plots=False  # 不画大图，只输出核心数据
        )

        # ----------------- 4. 格式化提取并打印核心指标 -----------------
        print("\n" + "=" * 50)
        print(" 焊接缺陷检测性能评估报告 (50张样本) ")
        print("=" * 50)

        # 目标定位（Bounding Box）指标
        print("[ 边界框检测指标 (Box) ]")
        print(f" -> 查准率 (Precision): {metrics.box.mp:.4f}")
        print(f" -> 查全率 (Recall):    {metrics.box.mr:.4f}")
        print(f" -> F1-Score:           {metrics.box.f1[0]:.4f}")
        print(f" -> mAP50:              {metrics.box.map50:.4f}")

        print("-" * 30)

        # 精密分割（Mask）指标
        print("[ 缺陷分割指标 (Mask) ]")
        print(f" -> 查准率 (Precision): {metrics.seg.mp:.4f}")
        print(f" -> 查全率 (Recall):    {metrics.seg.mr:.4f}")
        print(f" -> F1-Score:           {metrics.seg.f1[0]:.4f}")
        print(f" -> mAP50:              {metrics.seg.map50:.4f}")
        print("=" * 50)

    else:
        print(f"[ERROR] 未知的任务类型: '{task}'，请设置为 'train' 或 'val'")