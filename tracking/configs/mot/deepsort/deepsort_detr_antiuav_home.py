_base_ = [
    '../../_base_/models/deep_detr_base.py',
    '../../_base_/datasets/antiuav_full_home.py',
    '../../_base_/drone_runtime.py'
]
model = dict(
    type='DeepSORT',
    pretrains=dict(
        detector=  # noqa: E251
        # 'C:\\Users\\Matt\\Documents\\PhD\\drone\\mmtracking\\configs\\mot\\deepsort\\faster.pth', # noqa: E501
        "/hdd/PhD/drone/Strig-UAV-Project/anti-uav/DETR/full/latest.pth",
        # "/home2/lgfm95/drone/Strig-UAV-Project/anti-uav/DETR/full/latest.pth",
        reid=  # noqa: E251
        'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
    ))