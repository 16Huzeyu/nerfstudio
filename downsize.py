from PIL import Image
import os

# 定义文件夹路径
folder_path = 'data/ni4cam/masks'  # 将路径替换为实际文件夹的路径

# 遍历文件夹中的子文件夹
for subdir in os.listdir(folder_path):
    subdir_path = os.path.join(folder_path, subdir)
    
    # 遍历子文件夹中的图片文件
    for filename in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, filename)
        
        # 打开图片
        image = Image.open(file_path)
        
        # 获取原始分辨率
        width, height = image.size
        
        # 计算新的分辨率（宽度和高度减半）
        new_width = width // 2
        new_height = height // 2
        
        # 调整图片大小
        resized_image = image.resize((new_width, new_height))
        
        # 覆盖原始图片
        resized_image.save(file_path)
