import os
import random
from pathlib import Path
from ultralytics import YOLO


def predict_with_fixed_labels():
    # 1. 定义路径
    model_path = Path(r"D:\DataSets\InsideMachine\Radiographsweldingdefectdetection\m60.pt")
    image_dir = Path(r"D:\DataSets\InsideMachine\Radiographsweldingdefectdetection\train\images")

    if not model_path.exists() or not image_dir.exists():
        print("❌ 错误：请检查模型或图片路径是否正确！")
        return

    # 2. 直接根据【编号 0-12】定义英语标签映射
    id_to_en = {
        0: "Porosity",
        1: "Inclusion",
        2: "Under-cut",
        3: "Burn-through",
        4: "Crack",
        5: "Overlap",
        6: "Reference 1",
        7: "Reference 2",
        8: "Reference 3",
        9: "Hidden porosity",
        10: "Shrinkage cavity",
        11: "Lack of fusion",
        12: "Incomplete root penetration"
    }

    # 3. 获取并随机抽取 50 张图片
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [str(p) for p in image_dir.iterdir() if p.suffix.lower() in valid_extensions]

    total_images = len(image_files)
    if total_images == 0:
        print("❌ 错误：目录中没有找到有效的图片。")
        return

    sample_size = min(50, total_images)
    selected_images = random.sample(image_files, sample_size)
    print(f"📂 检测到 {total_images} 张图片，已随机抽取 {sample_size} 张进行预测...")

    # 4. 加载 YOLO 权重（不做任何危险的底层修改）
    print("⏳ 正在加载 YOLO 权重...")
    model = YOLO(model_path)

    # 5. 开始推理预测（注意：这里设置 save=False，不让它用默认乱码标签保存）
    print("🚀 开始预测...")
    results = model.predict(
        source=selected_images,
        save=False,  # 关键：不让框架自动保存乱码图
        conf=0.25,  # 置信度阈值
        device='cpu'  # 可改为 '0' 提升速度
    )

    # 6. 【终极修复】在保存前，强行将每个结果对象的标签字典改为英文
    output_dir = Path("runs/segment/predict_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🎨 正在渲染英文标签并保存图片...")
    import cv2

    for result in results:
        # 强行覆盖当前结果对象内部的名称字典
        result.names = id_to_en

        # 此时 plot() 会自动读取上面刚刚改好的英文名字字典
        annotated_frame = result.plot()

        # 获取原文件名并拼接保存路径
        img_name = Path(result.path).name
        save_path = output_dir / img_name

        # 保存图片
        cv2.imwrite(str(save_path), annotated_frame)

    print("\n🎉 预测并保存成功！")
    print(f"📊 带有清晰英文标签的图片已完美保存至: {output_dir.resolve()}")


if __name__ == "__main__":
    predict_with_fixed_labels()