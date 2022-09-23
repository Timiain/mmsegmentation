_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1.py'
]
model=dict(
    decode_head=dict(
        num_classes=6,
        loss_decode=dict(
            type='CrossEntropyContrastMMALoss', use_sigmoid=False, loss_weight=1.0,
            # used this class weight for vaihingen
            class_weight=[1, 1, 1, 1, 1 , 1])))
