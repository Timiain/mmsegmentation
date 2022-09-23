_base_ = './fcn_hr18_4x4_512x512_80k_vaihingen.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        type='FCNHeadWithEmb',
        in_channels=[48, 96, 192, 384], 
        channels=sum([48, 96, 192, 384]),
        input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        norm_cfg=norm_cfg,
        align_corners=False,
        num_classes=6,
        loss_decode=dict(
            type='CrossEntropyContrastMMALoss', use_sigmoid=False, loss_weight=1.0,
            # used this class weight for vaihingen
            class_weight=[1,1,1,1,1,1]
        )
    )
)
