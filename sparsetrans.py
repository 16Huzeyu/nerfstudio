import json

# 读取transformsdense.json文件
with open('/home/ubuntu/project/nerf/nerfstudio/data/baixiniu/leftfront/transformsdense.json', 'r') as file:
    data = json.load(file)

# 从frames数组中每五个元素中选择一个
selected_frames = data['frames'][::5]
new_data = {
    "camera_model": "OPENCV",
    'frames': selected_frames,

}
# 创建新的transforms.json文件并保存选中的帧
with open('/home/ubuntu/project/nerf/nerfstudio/data/baixiniu/leftfront/transforms.json', 'w') as file:
    json.dump(new_data, file)
