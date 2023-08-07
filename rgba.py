from PIL import Image
import os

# 设置输入文件夹和输出文件夹路径
input_folder = "data/sim/car10/images"
output_folder = "data/sim/car10/images1"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 确定文件的完整路径
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # 检查文件是否为图片
    if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        # 打开图像文件
        image = Image.open(input_path)

        # 转换为 RGBA 模式（如果不是）
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # 创建白色背景图像
        background = Image.new("RGBA", image.size, (255, 255, 255))

        # 将图像粘贴到白色背景上，透明区域将变为白色
        composite = Image.alpha_composite(background, image)

        # 转换为 RGB 模式
        rgb_image = composite.convert("RGB")

        # 保存处理后的图像
        rgb_image.save(output_path)
        print(f"Converted {filename}")

print("Conversion complete!")