_base_ = [
    '../_base_/models/fcn_hrbc18.py', '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(decode_head=dict(num_classes=6))
