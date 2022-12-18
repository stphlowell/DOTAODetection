_base_ = 'ssd512_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(num_classes=18),
    pretrained = None)

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'airport', 'helipad')
data = dict(
    train=dict(
        type = dataset_type,
        img_prefix='data/dota/train/',
        classes=classes,
        ann_file='data/dota/annotations/instances_train.json'),
    val=dict(
        type = dataset_type,
        img_prefix='data/dota/val/',
        classes=classes,
        ann_file='data/dota/annotations/instances_val.json'),
    test=dict(
        type = dataset_type,
        img_prefix='data/dota/test/',
        classes=classes,
        ann_file='data/dota/annotations/instances_valfull.json'),
    samples_per_gpu = 1,
    workers_per_gpu = 2)

data_root = 'data/dota/'
work_dir = 'ssd_dota'
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

# We can use the pre-trained SSD model to obtain higher performance
load_from = 'checkpoints/ssd512_coco_20210803_022849-0a47a1ca.pth'