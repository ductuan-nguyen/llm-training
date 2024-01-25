import subprocess

training_script = 'CUDA_VISIBLE_DEVICES=1 python train_trl.py \
    --dataset_path /home4/tuannd/llm-training/data/Data_Vi_QA_v1.1/QA_Uni/hust_no_ans.csv \
    --epochs 1 \
    --lr 2e-5 \
    --max_seq_length 4096 \
    --model_id vilm/vinallama-7b-chat \
    --hf_token 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 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --checkpoint_local_path ./viqauni_vinallama_v2 \
    --gradient_checkpointing'

subprocess.run(training_script, shell=True)