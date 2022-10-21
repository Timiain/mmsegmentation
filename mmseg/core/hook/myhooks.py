# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.runner import HOOKS, Hook
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as pp
import seaborn as sns; 
from matplotlib import cm
import torch
from torch import nn

train_flag = True 

experiment_name='swim_b_potsdam'
#experiment_name='vaihingen_mma_balance_norm_normx'

class ListRecorder(object):
    """A generate recodedr tool"""
    def __init__(self):
        self.data={}
    
    def append(self,key,value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def to_csv(self,name):
        #补全大小
        max =0
        for key in self.data:
            if len(self.data[key])>max:
                max = len(self.data[key])
        
        for key in self.data:
            diff =max -  len(self.data[key])
            for i in range(diff):
                self.data[key].append(-1)

        import pandas as pd
        df = pd.DataFrame.from_dict(self.data)
        df.to_csv(name)

class record_uint :
    def __init__(self):
        pass

    def add():
        pass
CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
               'car', 'clutter')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
               [255, 255, 0], [255, 0, 0]]
r_uint = record_uint()

def flush_train():
    r_uint.train_total_loss=None
    r_uint.train_gt_semantic_seg=None
    r_uint.train_seg_logit_scale=None
    r_uint.train_seg_logits = None

# trick in mmseg/models/decode_head.py
def trick_train(func):
    def wrapper(*args):
        #print('[DEBUG]: enter {}()'.format(func.__name__))
        train_flag=True
        losses,seg_logits,gt_semantic_seg,train_cfg,seg_logit_scale = func(*args)

        r_uint.train_seg_logits = seg_logits
        if 'loss_ce' in losses:
            total_loss = losses['loss_ce']
            mean = torch.mean(total_loss)
            losses['loss_ce'] = mean

            r_uint.train_total_loss=total_loss
            r_uint.train_gt_semantic_seg=gt_semantic_seg
            r_uint.train_seg_logit_scale=seg_logit_scale
        else:
            r_uint.train_gt_semantic_seg=gt_semantic_seg
            r_uint.train_seg_logit_scale=seg_logit_scale


        return losses
    return wrapper

iter_eval_recorder = ListRecorder()
# trick in mmseg/segmentors/encoder_decoder.py
def trick_encode_decode(func):
    def wrapper(*args):
        #print('[DEBUG]: enter {}()'.format(func.__name__))
        out_resize,out,x = func(*args)

        bsl= out.cpu().numpy()
        asl= out_resize.cpu().numpy()
        for i in range(len(CLASSES)):
            iter_eval_recorder.append('bsl_max_{}'.format(i),np.max(bsl[:,i,:,:]))
            iter_eval_recorder.append('bsl_mean_{}'.format(i),np.mean(bsl[:,i,:,:]))
            iter_eval_recorder.append('bsl_min_{}'.format(i),np.min(bsl[:,i,:,:]))
            iter_eval_recorder.append('asl_max_{}'.format(i),np.max(asl[:,i,:,:]))
            iter_eval_recorder.append('asl_mean_{}'.format(i),np.mean(asl[:,i,:,:]))
            iter_eval_recorder.append('asl_min_{}'.format(i),np.min(asl[:,i,:,:]))

        return out_resize
    return wrapper

# trick in mmseg/datasets/custom.py
def trick_pre_eval(func):
    def wrapper(*args, **kwargs):
        #print('[DEBUG]: enter {}()'.format(func.__name__))
        pre_eval_results,preds,seg_maps = func(*args, **kwargs)
        pred = preds[0]
        gt = seg_maps[0]
        gt = gt-1
        gt[gt==254]=5

        #print('[{}]'.format(np.unique(gt)))
        for i in range(len(CLASSES)):
            i_slice = (pred == i).astype(np.int)
            slice_park = i_slice*gt
            sum_slice = np.sum(i_slice)

            tmp = (gt==i).astype(np.int)
            iter_eval_recorder.append('gt_{}'.format(i),np.sum(tmp))

            #add 1,2,3,4,5
            for j in range(1,len(CLASSES)):
                j_count = np.sum((slice_park==j).astype(np.int))
                iter_eval_recorder.append('pred_{}_to_{}'.format(i,j), j_count)
                sum_slice -= j_count
            #add 0
            iter_eval_recorder.append('pred_{}_to_{}'.format(i,0), sum_slice)
        
        for i in range(len(CLASSES)):
            i_slice = (gt == i).astype(np.int)
            slice_park = i_slice*pred
            sum_slice = np.sum(i_slice)

            #add 1,2,3,4,5
            for j in range(1,len(CLASSES)):
                j_count = np.sum((slice_park==j).astype(np.int))
                iter_eval_recorder.append('gt_{}_to_{}'.format(i,j), j_count)
                sum_slice -= j_count
            #add 0
            iter_eval_recorder.append('gt_{}_to_{}'.format(i,0), sum_slice)

        return pre_eval_results
    return wrapper

import numpy as np
def collect_grad(prob, target):
    #softmax step gradient 
    pos_part = prob*prob*target
    #neg 
    _hat_prob = -1 * prob * (1-target)
    _slice = torch.sum(prob*target,dim=1).unsqueeze(1)
    neg_part = torch.abs(_slice * _hat_prob)
    
    pos_grad = torch.sum(pos_part, dim=1)
    neg_grad = torch.sum(neg_part, dim=1)
    return pos_grad,neg_grad

def expand_label(pred, gt_classes):
    shp_x = pred.shape
    shp_y = gt_classes.shape
    if len(shp_x) != len(shp_y):
        gt_classes = gt_classes.view((shp_y[0], 1, *shp_y[1:]))
    y_onehot = torch.zeros(shp_x)
    if gt_classes.device.type == "cuda":
        y_onehot = y_onehot.cuda(gt_classes.device.index)
    y_onehot.scatter_(1, gt_classes, 1)
    return y_onehot

@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, interval=50,by_epoch=False):
        self.interval = interval
        self.by_epoch = by_epoch
        self.iter_train_recorder = ListRecorder()
    
    def record_train_iter(self,r_uint):
        ignore_gt = r_uint.train_gt_semantic_seg==255
        
        r_uint.train_gt_semantic_seg[r_uint.train_gt_semantic_seg==255]=5
        gt = r_uint.train_gt_semantic_seg.cpu().numpy()
        ignore_gt = ignore_gt.cpu().numpy()

        for i in range(len(CLASSES)):
            labels = np.sum((gt==i).astype(np.int))
            if i != 5:
                self.iter_train_recorder.append('labelnum_{}'.format(i), labels)
            else:
                self.iter_train_recorder.append('labelnum_{}'.format(i), labels-np.sum(ignore_gt))
        
        
        m = nn.Softmax(dim=1)
        if not(isinstance(r_uint.train_seg_logit_scale,list) or isinstance(r_uint.train_seg_logit_scale,tuple)):
            logits = m(r_uint.train_seg_logit_scale)
        else:
            logits = m(r_uint.train_seg_logit_scale[0])

        onehot = expand_label(logits, r_uint.train_gt_semantic_seg)
        pos_grad,neg_grad = collect_grad(logits,onehot)
        onehot = onehot.cpu().numpy()
        pos_grad = pos_grad.detach().cpu().numpy()
        neg_grad = neg_grad.detach().cpu().numpy()


        for i in range(len(CLASSES)):
            if i != 5:
                class_mask = onehot[:,i,:,:]
            else:
                class_mask = onehot[:,i,:,:]*(1-ignore_gt)

            tmp1 =class_mask*pos_grad
            pos = np.sum(tmp1)
            tmp2 =class_mask*neg_grad
            neg = np.sum(tmp2)
            self.iter_train_recorder.append('gradent_pos_{}'.format(i),pos)
            self.iter_train_recorder.append('gradent_neg_{}'.format(i),neg)
        
     
    def before_run(self, runner):
        # for batch_indices, data in zip(loader_indices, data_loader):
        #     pass
        #print("before_train_iter")
        pass

    def before_train_epoch(self, runner):
        pass

    def before_val_epoch(self, runner):
        pass

    def before_train_iter(self, runner):
        #print("before_train_iter")
        flush_train()

    def after_train_iter(self, runner):
        #print("after_train_iter")
        #print(r_uint)
        
        self.record_train_iter(r_uint)

        if self.every_n_iters(runner, self.interval):
            assert torch.isfinite(runner.outputs['loss']), \
                runner.logger.info('loss become infinite or NaN!')
        flush_train()

    def after_run(self, runner):
        m = runner.model.module
        print(m)
        last = m.decode_head.conv_seg.weight.data.detach().cpu().numpy()
        class_num = len(m.CLASSES)
        #范数
        last = np.reshape(last,(class_num,-1))
        last_norms=np.linalg.norm(last,axis=1,keepdims=True)
        print(last_norms)
        #角度
        A = cosine_similarity(last, last) 
        print("cosine_similarity for each category logit vector")
        print(A)
        fig = pp.figure()
        ax1 = fig.add_subplot(111)
        cmap = cm.get_cmap('Blues', len(last))
        cmap.set_bad('w') # default value is 'k'
        ax1.imshow(A, cmap=cmap )
        # pp.ylabel("Cluster", size = 16)
        # pp.xlabel("Cluster", size = 16)
        sns.despine()
        pp.colorbar(ax1.matshow(A.T, cmap=cmap, vmin=-0.1, vmax=1), shrink=.75)

        #ax1.xaxis.set_label_position('bottom')
        #ax1.xaxis.set_ticks_position('bottom')
        pp.xticks(range(0,len(last),1), size = 16)
        pp.yticks(range(0,len(last),1), size = 16)
        #pp.show()
        prior_pth='last_layer'
        pp.savefig('./work_dirs/{}_cosine_similarity_{}.jpg'.format(experiment_name,prior_pth), dpi=220, bbox_inches='tight')

        self.iter_train_recorder.to_csv('./work_dirs/{}_iter_train_recorder.csv'.format(experiment_name))
        iter_eval_recorder.to_csv('./work_dirs/{}_iter_eval_recorder.csv'.format(experiment_name))


