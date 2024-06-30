# llama_2 7b lora models
for ((i=1;i <= 13; i++)) do
python main.py --model_name llama_2 --model_type 7b \
 --lora True --lora_model_dir /U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-llama2-7b-epoch8-v3 \
 --lora_model_name checkpoint-${i}000 \
 --lora_checkpoint_step ${i}
 done

 # llama_2 13b lora models
for ((i=1;i <= 30; i++)) do
python main.py --model_name llama_2 --model_type 13b \
 --lora True --lora_model_dir /U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-llama2-13b-epoch5-v2 \
 --lora_model_name checkpoint-${i}000 \
 --lora_checkpoint_step ${i}
 done

 # qwen_1.5 0.5b lora models
for ((i=1;i <= 26; i++)) do
python main.py --model_name qwen_1.5 --model_type 0.5b \
 --lora True --lora_model_dir /U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-qwen1.5-0.5b-epoch15-v1 \
 --lora_model_name checkpoint-${i}000 \
 --lora_checkpoint_step ${i}
 done

# qwen_1.5 1.8b lora models
for ((i=1;i <= 25; i++)) do
python main.py --model_name qwen_1.5 --model_type 1.8b \
 --lora True --lora_model_dir /U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-qwen1.5-1.8b-epoch15-v1 \
 --lora_model_name checkpoint-${i}000 \
 --lora_checkpoint_step ${i}
 done

# qwen_1.5 4b lora models
for ((i=1;i <= 24; i++)) do
python main.py --model_name qwen_1.5 --model_type 4b \
 --lora True --lora_model_dir /U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-qwen1.5-4b-epoch15-v1 \
 --lora_model_name checkpoint-${i}000 \
 --lora_checkpoint_step ${i}
 done

# qwen_1.5 7b lora models
for ((i=1;i <= 24; i++)) do
python main.py --model_name qwen_1.5 --model_type 7b \
 --lora True --lora_model_dir /U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-qwen1.5-7b-epoch15-v1 \
 --lora_model_name checkpoint-${i}000 \
 --lora_checkpoint_step ${i}
 done

# qwen_1.5 14b lora models
for ((i=1;i <= 24; i++)) do
python main.py --model_name qwen_1.5 --model_type 14b \
 --lora True --lora_model_dir /U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-qwen1.5-14b-epoch15-v1 \
 --lora_model_name checkpoint-${i}000 \
 --lora_checkpoint_step ${i}
 done