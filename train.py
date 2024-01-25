import subprocess

subprocess.run(['bash', 'setup.sh'])

training_script = 'accelerate launch \
    --config_file fsdp_config.yaml \
    --num_processes 8 \
    train.py \
    --dataset_path /opt/ml/input/data/training \
    --epochs 3 \
    --lr 2e-5 \
    --model_id vilm/vinallama-7b-chat \
    --hf_token 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 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --checkpoint_local_path /opt/ml/checkpoints \
    --gradient_checkpointing'

subprocess.run(training_script, shell=True)