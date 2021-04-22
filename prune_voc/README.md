This codebase is derived from [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)
## Installation
### Prerequisites
- Python 3.7
- CUDA 8.0
- PyTorch 1.0.1

Create directory `data`
```
mkdir data
cd data
```
Refer to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) for installation of VOC datasets in `data` directory.

Install the CoCO API.
```
cd data
git clone https://github.com/pdollar/coco.git 
cd coco/PythonAPI
make
cd ../../..
```
Install all the python dependencies using pip:
```
pip install -r requirements.txt
```
Compile all the cuda dependencies.
```
cd lib
python setup.py build install
```


## File structure

[trainval_net.py](trainval_net.py) used for direct pruning while [trainval_net_in_pretrain.py](trainval_net_in_pretrain.py) is used for training ImageNet transferred ticket. [test_net.py](test_net.py) and [test_net_in_pretrain.py](test_net_in_pretrain.py) are the corresponding evaluation scripts. `lib/model/faster_rcnn/lth.py` contains the helper functions for applying LTH on the object detection models.

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
Save ImageNet trained ticket in in_weights/model.pth. Example run command for training 80% ImageNet transferred ticket on machine with multiple GPUs.
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
