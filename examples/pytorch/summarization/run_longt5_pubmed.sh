#!/bin/bash
DEBUG=0

DEBUG_CHECK=$([[ $DEBUG -eq 1 ]] && echo "on" || echo "off")
echo "Debug mode is $DEBUG_CHECK."

N_GPU=2
MODEL=$([[ $DEBUG -eq 1 ]] && echo "Stancld/LongT5-Local-Base" || echo "Stancld/LongT5-TGlobal-Large")
LAUNCH=$([[ $DEBUG -eq 1 ]] && echo "" || echo "-m torch.distributed.launch --nproc_per_node=$N_GPU")  # 2 GPUs given
BF16=$([[ $DEBUG -eq 1 ]] && echo "False" || echo "True")  # BF16 can be used on A100
NO_CUDA=$([[ $DEBUG -eq 1 ]] && echo "True" || echo "False")  # BF16 can be used on A100


TOTAL_BATCH_SIZE=$([[ $DEBUG -eq 1 ]] && echo 1 || echo 128)
PER_DEVICE_BATCH_SIZE=2
ACC_STEP=$((TOTAL_BATCH_SIZE / PER_DEVICE_BATCH_SIZE))
ACC_STEP=$([[ $DEBUG -eq 1 ]] && echo $ACC_STEP || echo $((ACC_STEP / N_GPU)))


SCRIPT="python $LAUNCH run_summarization.py \
    --model_name_or_path $MODEL \
    --do_train \
    --do_eval \
    --do_predict \
    --dataset_name ccdv/pubmed-summarization \
    --max_source_length 16384 \
    --max_target_length 512 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $ACC_STEP \
    --optim adafactor \
    --learning_rate 0.001 \
    --lr_scheduler_type constant \
    --num_train_epochs 20 \
    --gradient_checkpointing \
    --bf16=$BF16 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate \
    --generation_num_beams 1 \
    --generation_max_length 512 \
    --output_dir /tmp/longt5_pubmed \
    --report_to all \
    --logging_steps 10 \
    --eval_steps 500 \
    --no_cuda=$NO_CUDA
"

echo "Following script is going to be run: $SCRIPT"
eval $SCRIPT
