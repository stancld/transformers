DEBUG=1
N_GPU=2

DEBUG_CHECK=$([[ $DEBUG -eq 1 ]] && echo "on" || echo "off")
echo "Debug mode is $DEBUG_CHECK."

MODEL=$([[ $DEBUG -eq 1 ]] && echo "Stancld/LongT5-Local-Base" || echo "Stancld/LongT5-TGlobal-Large")


LAUNCH=$([[ $DEBUG -eq 1 ]] && echo "" || echo "-m torch.distributed.launch --nproc_per_node=$N_GPU")
BF16=$([[ $DEBUG -eq 1 ]] && echo "False" || echo "True")


TOTAL_BS=128
BS=1
ACC_STEP=$((TOTAL_BS / BS))


SCRIPT="python $LAUNCH run_summarization.py \
    --model_name_or_path $MODEL \
    --do_train \
    --do_eval \
    --do_predict \
    --dataset_name ccdv/pubmed-summarization \
    --source_prefix summarize \
    --max_source_length 16384 \
    --max_target_length 512 \
    --per_device_train_batch_size $BS \
    --gradient_accumulation_steps $ACC_STEP \
    --optim adafactor \
    --learning_rate 0.001 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --bf16=$BF16 \
    --per_device_eval_batch_size 8 \
    --output_dir /tmp/longt5_pubmed
"

echo "Following script is going to be run: $SCRIPT"
eval $SCRIPT
