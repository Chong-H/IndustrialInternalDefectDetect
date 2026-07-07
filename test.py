import os
from glob import iglob
from ultralytics import YOLO

# 1. 加载模型
model_path = r"D:\DataSets\InsideMachine\Radiographsweldingdefectdetection\m60.pt"
model = YOLO(model_path)

# 2. 指定输入图片目录
image_dir = r"D:\DataSets\InsideMachine\Radiographsweldingdefectdetection\train\images"
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

print("正在扫描并按名字顺序筛选前 80 张图片...")
all_image_paths = []
for path in iglob(os.path.join(image_dir, "*")):
    if path.lower().endswith(valid_extensions):
        all_image_paths.append(path)

# 按名字排序并取前 80 张
all_image_paths.sort()
selected_images = all_image_paths[:80]

if not selected_images:
    print(f"在目录 {image_dir} 中没有找到图片，请检查路径。")
else:
    # 🎯 定义保存结果的自定义输出目录
    output_dir = r"runs\segment\predict_bak"
    os.makedirs(output_dir, exist_ok=True)

    # 英文标签映射，防止图片中画框出现豆腐块
    english_names = {
        0: "QiKong_Porosity", 1: "JiaZa_Inclusion", 2: "YaoBian_Undercut",
        3: "ShaoChuan_BurnThrough", 4: "LieWen_Crack", 5: "HanLiu_Overlap",
        6: "Standard_1", 7: "Standard_2", 8: "Standard_3",
        9: "YinCangQiKong", 10: "AoXian_Cavity", 11: "WeiRongHe_NoFusion",
        12: "WeiHanTou_NoPenetration"
    }
    if hasattr(model, "model") and hasattr(model.model, "names"):
        model.model.names = english_names
    if hasattr(model, "_names"):
        model._names = english_names

    print(f"开始批量推理，并将结果另存为 '原名_bak.图片格式'...\n")

    # 3. 运行推理（注意：这里 save=False，不让 YOLO 用它自带的默认名字乱存）
    results = model.predict(source=selected_images, save=False, conf=0.25, verbose=False)

    # 4. 🎯 【核心修改】遍历推理结果，手动改名并保存
    for result in results:
        # 获取原始文件的绝对路径、基础名字和后缀名
        orig_path = result.path
        base_name = os.path.basename(orig_path)  # 比如: 0-341.jpg
        name_without_ext, ext = os.path.splitext(base_name)  # 拆分成: ('0-341', '.jpg')

        # 拼装出你需要的 xbaks.img 格式名字（如: 0-341_bak.jpg）
        bak_filename = f"{name_without_ext}_bak{ext}"
        save_path = os.path.join(output_dir, bak_filename)

        # 调用 result.save()，传入我们自己拼好的新路径
        result.save(filename=save_path)
        print(f"原图: {base_name} --> 已保存为: {bak_filename}")

    print(f"\n全部推理并更名完成！请去 {output_dir} 文件夹查看结果。")