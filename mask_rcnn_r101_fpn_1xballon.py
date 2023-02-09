# model settings
# _base_ = ['./mask_rcnn_r50_fpn_fp16_1x_coco.py']

_base_ = './mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
    )
)



# dataset settings
dataset_type = 'CocoDataset'
data_root = r'D:\workspace\flower_data\flower\balloon_dataset\balloon'
CLASSES = ('balloon',)

data = dict(
  train=dict(
      ann_file=data_root + '/train.json',
      img_prefix=data_root + '/train/',
      classes=CLASSES
  ),
  val=dict(
      ann_file=data_root + '/val.json',
      img_prefix=data_root + '/val/',
      classes=CLASSES
  ),
  test=dict(
      ann_file=data_root + '/val.json',
      img_prefix=data_root + '/val/',
      classes=CLASSES
  )
)

# evaluation = dict(metric=['bbox', 'segm'])
evaluation = dict(interval=1, metric=['bbox', 'segm'], save_best='segm_mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=30)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(interval=1,hooks=[dict(type='TextLoggerHook')])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = r'D:\workspace\flower_data\flower\balloon_dataset\mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'
resume_from = None

