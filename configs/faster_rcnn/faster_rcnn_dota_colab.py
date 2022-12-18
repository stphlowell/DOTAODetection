_base_ = 'faster_rcnn_r50_fpn_1x_coco.py'

# Change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=18)),
    backbone=dict(
        with_cp=True),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                gpu_assign_thr=1
            )
        ),
        rcnn=dict(
            assigner=dict(
                gpu_assign_thr=1
            )
        )
    )
)

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'airport', 'helipad')
data = dict(
    train=dict(
        type = dataset_type,
        img_prefix='DOTA_YOLOv5-6/train/',
        classes=classes,
        ann_file='DOTA_YOLOv5-6/train/_annotations.coco.json'),
    val=dict(
        type = dataset_type,
        img_prefix='DOTA_YOLOv5-6/val/',
        classes=classes,
        ann_file='DOTA_YOLOv5-6/val/_annotations.coco.json'),
    test=dict(
        type = dataset_type,
        img_prefix='DOTA_YOLOv5-6/test/',
        classes=classes,
        ann_file='DOTA_YOLOv5-6/test/_annotations.coco.json'),
    samples_per_gpu = 1,
    workers_per_gpu = 2)

data_root = 'data/dota/'
work_dir = 'faster_rcnn_dota'
optimizer = dict(
    lr = 0.002/8)
lr_config = dict(
    warmup = None)
log_config = dict(
    interval = 10)

evaluation=dict(
    metric='bbox',
    interval=1,
    classwise=True,
    metric_items=['mAP','mAP_50','mAP_75','mAP_s','mAP_m','mAP_l','AR@100','AR@300','AR@1000','AR_s@1000','AR_m@1000','AR_l@1000'],
    save_best='auto'
)

# Use the pre-trained Faster R-CNN model to obtain higher performance
load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'