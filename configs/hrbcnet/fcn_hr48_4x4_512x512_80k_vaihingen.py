_base_ = './fcn_hr18_4x4_512x512_80k_vaihingen.py'
model = dict(
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384))))
)
