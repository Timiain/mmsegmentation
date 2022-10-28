_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/vaihingen_mod1.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k_mod1.py'
]
model = dict(decode_head=dict(num_classes=6))