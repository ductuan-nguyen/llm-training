import subprocess

training_script = 'accelerate launch \
    --config_file fsdp_config.yaml \
    --num_processes 4 \
    train.py \
    --dataset_path sample_final \
    --epochs 1 \
    --lr 2e-5 \
    --model_id trick4kid/PhoGPT-7B5-Instruct-Patch \
    --hf_token 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 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --checkpoint_local_path ./checkpoints_phogpt_pack \
    --gradient_checkpointing'


subprocess.run(training_script, shell=True)