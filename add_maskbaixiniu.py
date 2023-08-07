import json

# 读取 transforms.json 文件
with open('/home/ubuntu/project/nerf/nerfstudio/data/ni4cam/transforms.json', 'r') as f:
    data = json.load(f)

# 遍历 frames 数组
for frame in data['frames']:
    # 获取 file_path 值
    file_path = frame['file_path']

    value = "masks"

    # 添加 "v" 字段，并赋值为提取的字符串
    frame['mask_path'] =  file_path.replace('images', value)+".png"

    # 替换 images 为提取的字符串


# 保存修改后的数据到新的文件
with open('/home/ubuntu/project/nerf/nerfstudio/data/ni4cam/transforms.json', 'w') as f:
    json.dump(data, f, indent=2)


# import os
# import shutil

# # 指定文件夹路径
# folder_path = "/home/ubuntu/project/nerf/nerfstudio/data/baixiniu/leftfront/masks_22"
# folder_path2 = "/home/ubuntu/project/nerf/nerfstudio/data/baixiniu/leftfront/masks_2"

# # 获取文件夹中的所有文件，并按文件名排序
# files = sorted(os.listdir(folder_path))

# # 遍历文件，并重命名为frames_0000x.jpg
# for i, file_name in enumerate(files):
#     # 检查文件是否为图片文件
#     if file_name.endswith((".jpg", ".jpeg", ".png")):
#         # 构建新的文件名
#         new_file_name = f"frame_{i+1:05}.jpg"
        
#         # 构建旧文件路径和新文件路径
#         old_file_path = os.path.join(folder_path, file_name)
#         new_file_path = os.path.join(folder_path2, new_file_name)
        
#         # 重命名文件
#         shutil.move(old_file_path, new_file_path)
