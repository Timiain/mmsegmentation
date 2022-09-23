## 合并更改文件部分
1. 评价指标 ./mmseg/core/evaluation
2. 添加了监视模型的hook ./mmseg/core/hook/myhooks
3. 损失函数部分 ./mmseg/models/losses/cross_entropy_loss
4. decode head ./mmseg/models/decode_heads/decode_head.py +fcn_head.py
5. 其他小修改可能没有收录

## 目前的基线运行方法
### 0. 准备数据

A. clone 

B. 数据拷贝
使用scp  从208的机器中拷贝，下面两个文件夹包含了所需要的压缩包 
```
./disk1/lhl/workspace/segv2/mmsegmentation/data/Vaihingen
./disk1/lhl/workspace/segv2/mmsegmentation/data/Potsdam
```
C. 或者从官网下载

https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx

D. 参考对应的转换工具进行转换
```
python ./tools/convert_datasets/vaihingen.py ./data/Vaihingen/

python ./tools/convert_datasets/potsdam.py ./data/Potsdam/
```


### 1. 激活空间
例如： conda activate /disk1/lhl/env/open-mmlab-seg/

### 2. 选择配置文件

自定义配置文件
```
nohup python ./tools/train.py ./configs/hrnet/fcn_hr48_4x4_512x512_80k_vaihingen_mma_balance_norm.py --seed 42 --gpu-id 1 > ./work_dirs/vaihingen_mma_balance_norm.txt &
```

默认配置文件
```
nohup python ./tools/train.py ./configs/hrnet/fcn_hr48_4x4_512x512_80k_vaihingen.py --seed 42 --gpu-id 0 > ./work_dirs/vaihingen.txt &
```
