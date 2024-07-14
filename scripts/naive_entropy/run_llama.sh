DATA_SET=Xsum
LOG_MODE=offline
MODEL_NAME=llama_2
# python compute_naive_entropy.py --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --model_type 7b
# python compute_naive_entropy.py --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --model_type chat_7b
# python compute_naive_entropy.py --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --model_type 13b
# python compute_naive_entropy.py --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --model_type chat_13b
python compute_naive_entropy.py --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --model_type 70b
# python compute_naive_entropy.py --model_name $MODEL_NAME --dataset $DATA_SET --wandb_mode $LOG_MODE --model_type chat_70b
