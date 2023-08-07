import json
import os
# 读取transforms.json文件
with open("/home/ubuntu/project/nerf/nerfstudio/data/nuscenes/transforms.json", "r") as file:
    data = json.load(file)

# 遍历frames数组中的每个对象
for frame in data["frames"]:
    # 添加depth_file_path字段，并设置其值为file_path的值
    frame["depth_file_path"] = frame["file_path"].replace("images", "depth")
    frame["depth_file_path"]=os.path.splitext(frame["depth_file_path"])[0] + '.png'
# 保存修改后的数据到transforms_modified.json文件
with open("/home/ubuntu/project/nerf/nerfstudio/data/nuscenes/transforms.json", "w") as file:
    json.dump(data, file, indent=2)

print("transforms_modified.json文件保存成功")
