#!/bin/sh
export CUDA_VISIBLE_DEVICES=3
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
#export GLOG_vmodule=operator=5
#export GLOG_v=10


python -u ./pdseg/train.py --log_steps 10 --cfg configs/cityscape.yaml --use_gpu --use_mpio \
DATASET.SEPARATOR " " \
DATASET.VPS True \
DATASET.NUM_CLASSES 2 \
TRAIN.PRETRAINED_MODEL_DIR "pretrain/mobilenet_cityscapes" \
TRAIN.MODEL_SAVE_DIR "snapshots/mobilenet_cityscape/v2_only_building_e100/" \
MODEL.DEEPLAB.BACKBONE "mobilenet" \
MODEL.FP16 False \
MODEL.SCALE_LOSS "dynamic" \
MODEL.ICNET.DEPTH_MULTIPLIER 1.0 \
MODEL.DEEPLAB.ASPP_WITH_SEP_CONV True \
MODEL.DEEPLAB.DECODER_USE_SEP_CONV True \
MODEL.DEEPLAB.ENCODER_WITH_ASPP False \
MODEL.DEEPLAB.ENABLE_DECODER False \
MODEL.DEFAULT_NORM_TYPE "bn" \
TRAIN.SYNC_BATCH_NORM True \
SOLVER.LR 0.01 \
TRAIN.SNAPSHOT_EPOCH 1 \
SOLVER.NUM_EPOCHS 100 \
SOLVER.LR_POLICY "poly" \
SOLVER.OPTIMIZER "sgd" \
BATCH_SIZE 16 \
DATALOADER.NUM_WORKERS 8 \
DATALOADER.BUF_SIZE 256
# --use_mpio MODEL.SCALE_LOSS "dynamic" \ TRAIN.PRETRAINED_MODEL "pretrain/coco_pretrained_0.4632/" \ MODEL.DEEPLAB.BACKBONE "mobilenet" \