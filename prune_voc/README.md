# LTH for Object Recognition
This repo is the official implementation for the CVPR submission **#10479**

## Installation
Refer to https://github.com/jwyang/faster-rcnn.pytorch for installation instructions.

## File structure

[trainval_net.py](trainval_net.py) used for direct pruning while [trainval_net_in_pretrain.py](trainval_net_in_pretrain.py) is used for training ImageNet transferred ticket. [test_net.py](test_net.py) and [test_net_in_pretrain.py](test_net_in_pretrain.py) are the corresponding evaluation scripts. [lib/model/faster_rcnn/lth.py] (lib/model/faster_rcnn/lth.py) contains the helper functions for applying LTH on the object detection models.

## Running experiments
### Direct pruning
Example run command for one-shot pruning 80% of network weights on machine with multiple GPUs.
```
 python trainval_net.py \
                    --dataset pascal_voc --net res18 \
                    --cuda --mGPUs --epochs 12\
                    --bs 12 --nw 10 --nr 2 \
                    --lri 600 --lr_warmup 600 --kp 0.2 \
                    --lr 1e-2 --lr_decay_step 10 \
                    --rs 0 --ml 1111111
```

### ImageNet transfer
Save ImageNet trained ticket in in_weights/model.pth. Example run command for training 80% ImageNet ticket on machine with multiple GPUs.
```
 python trainval_net.py \
                    --dataset pascal_voc --net res18 \
                    --cuda --mGPUs --epochs 12 \
                    --bs 12 --nw 10 --nr 1\
                    --lri 600 --lr_warmup 600 --kp 1.0 \
                    --lr 1e-2 --lr_decay_step 10 \
                    --rs 0 --ml 0000000 \
                    --pretrained_path in_weights/model.pth
```
