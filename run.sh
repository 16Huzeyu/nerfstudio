ns-render camera-path --load-config outputs/meetingroom1/nerfacto/2023-06-08_135437/config.yml --camera-path-filename data/polycam/meetingroom1/camera_paths/2023-06-08_135437.json --output-path renders/meetingroom1/2023-06-08_135437.mp4

ns-train nerfacto --data data/polycam/meetingroom2
ns-train nerfacto nuscenes-data  --cameras FRONT FRONT_LEFT FRONT_RIGHT


ns-process-data polycam --data data/polycam/7月18日下午5-19-poly.zip --output-dir data/polycam/yutian12 --max_dataset_size 1100

ns-process-data video --data data/willvideo/2023-06-03_14-46-36-front.mp4 --output-dir data/video/WILL --skip-colmap --num-frames-target 600
ns-process-data video --data data/willvideo/2023-06-03_14-46-36-front.mp4 --output-dir data/video/WILL --skip-colmap --num-frames-target 1500

ns-export tsdf --load-config outputs/meetingroom2/nerfacto/2023-06-08_164656/config.yml --output-dir mesh/meetingroom3

python nerfstudio/process_data/colmap2nerfstudio.py

ssh -L 48951:localhost:48951 ubuntu@124.222.35.220


python nerfstudio/scripts/datasets/process_nuscenes_masks.py --data_dir /mnt/cos/ML_data/nuscenes --output_dir /mnt/cos/ML_data/nuscenes/masks

ns-render camera-path --load-config  outputs/leftfront/nerfacto/2023-06-29_152552/config.yml --camera-path-filename data/baixiniu/leftfront/camera_paths/2023-06-29_112304.json --output-path renders/baixiniu/2023-06-30_17065511.mp4

ns-process-data metashape --data data/WILL/images3 --xml  data/WILL/4.xml --output-dir  data/nilong3 --num-downscales 0


ns-render camera-path --load-config outputs/leftfront/nerfacto/2023-07-13_121454/config.yml --camera-path-filename data/baixiniu/leftfront/camera_paths/2023-07-13_121454.json --output-path renders/leftfront/2023-07-13_121454.mp4

ns-viewer --load-config outputs/yutiandasha1f/nerfacto/2023-07-14_154055/config.yml