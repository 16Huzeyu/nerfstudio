import json

# 读取transforms.json文件
with open("/home/ubuntu/project/nerf/nerfstudio/data/ni4cam/transforms.json", "r") as file:
    data = json.load(file)

# 在每个元素的file_path开头为"images/cam_fr"时添加所需的键值对
for frame in data["frames"]:
    file_path = frame["file_path"]
    if file_path.startswith("images/cam_fl"):
        frame.update({
            # "w": 1920,
            # "h": 1080,
            # "fl_x": 1001.6565459617234,
            # "fl_y": 951.70259155016731,
            # "cx": 960.0,
            # "cy": 540.0,
            # "k1": -0.23754650175592318,
            # "k2": 0.040651713622527973,
            # "p1": 0.00048804160659301658,
            # "p2": 0.0032770959377238572,
            "mask_path": "masks/nomask.png"
        })

# 将修改后的数据保存回transforms.json文件
with open("/home/ubuntu/project/nerf/nerfstudio/data/ni4cam/transforms.json", "w") as file:
    json.dump(data, file, indent=4)
