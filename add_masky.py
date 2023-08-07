import json

# 读取 transforms.json 文件
with open('/home/ubuntu/project/nerf/nerfstudio/data/WILL/transforms.json', 'r') as f:
    data = json.load(f)

# 遍历 frames 数组
for frame in data['frames']:
    # 获取 file_path 值
    file_path = frame['file_path']

    # 提取两个 __ 之间的字符串
    start_index = file_path.find('__') + 2
    end_index = file_path.find('__', start_index)
    #value = "/mnt/cos/ML_data/nuscenes/masks/masks/"+file_path[start_index:end_index]
    value = "segs"
    # 添加 "v" 字段，并赋值为提取的字符串
    # frame['mask_path'] =  file_path.replace('images', value)+".png"
    frame['mask_path'] =  file_path.replace('images', value)
    #frame['mask_path'] =  "/home/ubuntu/project/nerf/nerfstudio/data/baixiniu/leftfront/masks_2/1111.png"
    # 替换 images 为提取的字符串


# 保存修改后的数据到新的文件
with open('/home/ubuntu/project/nerf/nerfstudio/data/WILL/transforms.json', 'w') as f:
    json.dump(data, f, indent=2)
