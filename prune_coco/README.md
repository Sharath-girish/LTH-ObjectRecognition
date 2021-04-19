# LTH for object recognition
This repo is the official implementation for the CVPR submission **#10479**

Note: This repo is built upon Facebook's open source detectron2 library. hence we have retained the copyright and the author notices from the original repo. We’ll update these when making the repo public.

# Installation
For installing the required dependecies to run this repo, please refer to the [detectron_README](detectron_README.md) document provided by detectron2 and follow the steps. 

# File Structure
The different parts/experiments require switching different options in this repository through the ```MODE``` flag in config. The general structure is as follows:
```tools/train_net.py ```:  The main python script to execute for all forms of training and eval. 
```tools/lth.py``` : The LTH class which controls the functions relating to applying LTH on the object recognition models. 
```detectron2/configs/defaults.py```: Contains details about the various parameters that needs to be passed while running ```train_net.py```

# Running experiments
## Direct pruning
Example way to run: 
Here we are training a keypoint detector model with Resnet18 backbone and  with 40% sparsity using direct pruning (all layers are pruned). 
Additional configs are found in ```configs/``` directory.

```
 python tools/train_net.py --num-gpus 4 --resume \
   --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
 	SOLVER.BASE_LR 0.015 \
 	SOLVER.WARMUP_ITERS 2000 \
 	 SOLVER.WARMUP_FACTOR 5e-4 \
 	LOTTERY_KEEP_PERCENTAGE 0.6 \
 	NUM_ROUNDS 2 \
 	OUTPUT_DIR temp/
 	MODE lottery
```


## Transfer tickets
Example:
To train a Resnet-18 keypoint detector with pruned imagnet backbone (transferred) at 90% sparsity. 

```
python tools/train_net.py --num-gpus 2 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	NUM_ROUNDS 2 \
	IMAGENET_TICKET resources/resnet_18_ticket_10.pth \
	IMAGENET_TICKET_TYPE res18 \
	OUTPUT_DIR temp/ \
	MODE transfer_imagenet 
```
## Pruning only backbone
Here we report numbers related to directly pruning only the backbone of the network, for both detection and keypoint estimation tasks. 
The command below trains a resnet18 keypoint model with **50% sparsity in its backbone network**.
```
python tools/train_net.py --num-gpus 2 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4 \
	LOTTERY_KEEP_PERCENTAGE 0.5 \
	NUM_ROUNDS 2 \
	OUTPUT_DIR temp/  \
	LATE_RESET_CKPT  resources/keypoint_r18_fpn_base/model_0003534.pth \
	MODEL.WEIGHTS  resources/keypoint_r18_fpn_base/model_final.pth \
	TEST.COMPUTE_FLOPS True \
	MODE prune_backbone
```
In this branch, we can also set ```TEST.COMPUTE_FLOPS``` argument to ```True``` to obtain flops and memory of the model. 

## Mask transfer
To run experiments where only mask (not values) is transferred from imagenet models. 
Example: Training a resnet50 backbone detector where the backbone mask is transferred with 90% sparsity. 
```
python tools/train_net.py --num-gpus 2 --resume \
      --config-file  configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	SOLVER.WARMUP_ITERS 2000 \
	 SOLVER.WARMUP_FACTOR 5e-4\
	LOTTERY_KEEP_PERCENTAGE 0.01\
	NUM_ROUNDS 2\
	OUTPUT_DIR temp/ \
	LATE_RESET_CKPT  resources/mask_r50_fpn_base/model_0007329.pth \
	MODEL.WEIGHTS  resources/mask_r50_fpn_base/model_final.pth \
	MODE mask_transfer

```
## Task Transfer
To run experiments relating to transferring masks across __tasks__. 
Example:
Code to train a Res18 keypoint detector with 80% sparsity, with the mask being obtained from the corresponding detection task. 
```
python tools/train_net.py --num-gpus 2 --resume \
  --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
	SOLVER.BASE_LR 0.015 \
	SOLVER.WARMUP_ITERS 5000 \
	 SOLVER.WARMUP_FACTOR 2e-4 \
	NUM_ROUNDS 2 \
	OUTPUT_DIR temp/ \
	SOURCE_TASK det \
	SOURCE_MODEL  output/mask_rcnn_r18_prune_20_late/model_0007329.pth \
	MODE task_transfer
```

