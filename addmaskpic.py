from PIL import Image
import os

# 设置输入文件夹和输出文件夹路径
input_folder = "data/sim/car5/images3"
output_folder = "data/sim/car5/mask_2"

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

        # 创建新图像对象，大小和原图像相同，深度为8位（L模式）
        new_image = Image.new("L", image.size)

        # 遍历每个像素
        for x in range(image.width):
            for y in range(image.height):
                # 获取当前像素的 RGBA 值
                r, g, b, a = image.getpixel((x, y))

                # 判断像素的透明度，将透明区域变为黑色，非透明区域变为白色
                if a < 128:  # 透明度小于128，即认为是透明区域
                    new_image.putpixel((x, y), 0)  # 将透明区域设为黑色
                else:
                    new_image.putpixel((x, y), 255)  # 将非透明区域设为白色

        # 保存处理后的图像
        new_image.save(output_path)
        print(f"Converted {filename}")

print("Conversion complete!")
import json

# 读取 transforms.json 文件
with open('/home/ubuntu/project/nerf/nerfstudio/data/sim/car5/transforms.json', 'r') as f:
    data = json.load(f)

# 遍历 frames 数组
for frame in data['frames']:
    # 获取 file_path 值
    file_path = frame['file_path']

    value = "mask_2"

    # 添加 "v" 字段，并赋值为提取的字符串
    frame['mask_path'] =  file_path.replace('images', value)

    # 替换 images 为提取的字符串


# 保存修改后的数据到新的文件
with open('/home/ubuntu/project/nerf/nerfstudio/data/sim/car5/transforms.json', 'w') as f:
    json.dump(data, f, indent=2)