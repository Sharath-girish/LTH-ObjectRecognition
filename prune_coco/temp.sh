export DETECTRON2_DATASETS=/fs/vulcan-datasets/

#Direct Pruning.
# python tools/train_net.py --num-gpus 2 --resume \
#    --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
#  	SOLVER.BASE_LR 0.015 \
#  	SOLVER.WARMUP_ITERS 2000 \
#  	 SOLVER.WARMUP_FACTOR 5e-4 \
#  	LOTTERY_KEEP_PERCENTAGE 0.6 \
#  	NUM_ROUNDS 2 \
#  	OUTPUT_DIR temp/ \
#  	LATE_RESET_CKPT  resources/keypoint_r18_fpn_base/model_0003534.pth \
# 	MODE lottery \
# 	MODEL.WEIGHTS  resources/keypoint_r18_fpn_base/model_final.pth #Ckpt to reset from.

# #Transfer Tickets
# python tools/train_net.py --num-gpus 2 --resume \
#   --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 2000 \
# 	 SOLVER.WARMUP_FACTOR 5e-4 \
# 	NUM_ROUNDS 2 \
# 	IMAGENET_TICKET resources/resnet_18_ticket_10.pth \
# 	IMAGENET_TICKET_TYPE res18 \
# 	OUTPUT_DIR temp/ \
# 	MODE transfer_imagenet 

#Mask transfer
# python tools/train_net.py --num-gpus 2 --resume \
#       --config-file  configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
# 	SOLVER.WARMUP_ITERS 2000 \
# 	 SOLVER.WARMUP_FACTOR 5e-4\
# 	LOTTERY_KEEP_PERCENTAGE 0.01\
# 	NUM_ROUNDS 2\
# 	OUTPUT_DIR temp/ \
# 	LATE_RESET_CKPT  resources/mask_r50_fpn_base/model_0007329.pth \
# 	MODEL.WEIGHTS  resources/mask_r50_fpn_base/model_final.pth \
# 	MODE mask_transfer

#task transfer
# python tools/train_net.py --num-gpus 2 --resume \
#   --config-file configs/COCO-Keypoints/keypoint_rcnn_r18_FPN_1x.yaml \
# 	SOLVER.BASE_LR 0.015 \
# 	SOLVER.WARMUP_ITERS 5000 \
# 	 SOLVER.WARMUP_FACTOR 2e-4 \
# 	NUM_ROUNDS 2 \
# 	OUTPUT_DIR temp/ \
# 	SOURCE_TASK det \
# 	SOURCE_MODEL  output/mask_rcnn_r18_prune_20_late/model_0007329.pth \
# 	MODE task_transfer

#prune only backbone
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
