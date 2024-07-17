source scripts/reset.sh

DATA_SET=Xsum
LOG_MODE=offline
SCRIPT=compute_block_entropy.py
MAX_NUM_INPUT_TOKENS=50
MAX_NEW_TOKENS=15
TRUNCATE=True

MODEL_NAME=qwen_1.5
model_types=(0.5b chat_0.5b 1.8b chat_1.8b 4b chat_4b 7b chat_7b 14b chat_14b 32b chat_32b 72b chat_72b)
for model_type in ${model_types[*]}; 
do 
    echo "python $SCRIPT --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --max_new_tokens $MAX_NEW_TOKENS --truncate $TRUNCATE --model_name $MODEL_NAME --model_type $model_type"
    python $SCRIPT --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --truncate $TRUNCATE --model_name $MODEL_NAME --model_type $model_type
done

# python $SCRIPT --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --model_type 0.5b
# python $SCRIPT --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --model_type chat_0.5b
# python $SCRIPT --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --model_type 1.8b
# python $SCRIPT --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --model_type chat_1.8b
# python $SCRIPT --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --model_type 4b
# python $SCRIPT --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --model_type chat_4b
# python $SCRIPT --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --model_type 7b
# python $SCRIPT --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --model_type chat_7b
# python $SCRIPT --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --model_type 14b
# python $SCRIPT --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --model_type chat_14b
# python $SCRIPT --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --model_type 32b
# python $SCRIPT --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --model_type chat_32b
# python $SCRIPT --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --model_type 72b
# python $SCRIPT --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --max_num_input_tokens $MAX_NUM_INPUT_TOKENS --model_type chat_72b

MODEL_NAME=qwen_1
