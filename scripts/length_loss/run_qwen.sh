source scripts/reset.sh

DATA_SET=HC3
LOG_MODE=offline
SCRIPT=compute_length_loss.py

MODEL_NAME=qwen_1.5
model_types=(0.5b chat_0.5b 1.8b chat_1.8b 4b chat_4b 7b chat_7b 14b chat_14b 32b chat_32b 72b chat_72b)
for model_type in ${model_types[*]}; 
do 
    echo "python $SCRIPT --dataset $DATA_SET --wandb_mode $LOG_MODE --model_name $MODEL_NAME --model_type $model_type"
    python $SCRIPT --dataset $DATA_SET --wandb_mode $LOG_MODE --model_name $MODEL_NAME --model_type $model_type
done

MODEL_NAME=qwen_1
