MODEL_NAME=qwen_1.5
# python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type 0.5b
# python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type chat_0.5b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type 1.8b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type chat_1.8b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type 4b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type chat_4b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type 7b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type chat_7b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type 14b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type chat_14b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type 32b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type chat_32b

MODEL_NAME=qwen_1
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type 1.8b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type chat_1.8b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type 7b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type chat_7b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type 14b
python compute_naive_entropy.py --model_name $MODEL_NAME --wandb_mode online --model_type chat_14b