#!/bin/sh
export CUDA_VISIBLE_DEVICES=3
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

python -u ./pdseg/train.py --log_steps 10 --cfg configs/cityscape.yaml --use_gpu --use_mpio \
DATASET.SEPARATOR " " \
TRAIN.PRETRAINED_MODEL_DIR "snapshots/mobilenetv1_cityscape/no_aspp_decoder_coco_pretrained_e400_bs16_sync_lr0.01/400" \
TRAIN.MODEL_SAVE_DIR "snapshots/mobilenetv1_cityscape/no_aspp_decoder_coco_pretrained_e400_bs16_sync_lr0.01_2class/" \
MODEL.DEEPLAB.BACKBONE "mobilenetv1" \
MODEL.FP16 False \
MODEL.SCALE_LOSS "dynamic" \
DATASET.VPS True \
DATASET.NUM_CLASSES 2 \
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
# --use_mpio MODEL.SCALE_LOSS "dynamic" \ TRAIN.PRETRAINED_MODEL "pretrain/coco_pretrained_0.4632/" \ MODEL.DE
#TRAIN.PRETRAINED_MODEL_DIR "snapshots/mobilentv1_coco/e40_lr001_bs16/40" \
#TRAIN.MODEL_SAVE_DIR "snapshots/mobilenetv1_cityscape/no_aspp_decoder_coco_pretrained_e400_bs16_sync_lr0.01/" \

