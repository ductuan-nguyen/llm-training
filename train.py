import argparse

import torch
from datasets import load_from_disk
from huggingface_hub import login
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from transformers import (AutoConfig, AutoModelForCausalLM, Trainer, AutoTokenizer,
                          TrainingArguments, default_data_collator, set_seed)


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        help="Model id to use for training.",
    )

    parser.add_argument(
        "--dataset_path", type=str, default="/opt/ml/input/data/training", help="Path to dataset."
    )

    parser.add_argument(
        "--hf_token", type=str, default=None, help="Hugging Face token."
    )

    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to use for training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation step",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
    )

    parser.add_argument(
        "--checkpoint_local_path",
        type=str,
        default="/opt/ml/checkpoints",
    )
    args, _ = parser.parse_known_args()

    if args.hf_token:
        print(f"Logging into the Hugging Face Hub s...")
        login(token=args.hf_token)

    return args


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

    
def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
    
    
def training_function(args):
    # set seed
    set_seed(args.seed)

    print('Loading data...', end='')
    dataset = load_from_disk(args.dataset_path)
    print('dataset size: ', len(dataset))
    print_gpu_utilization()
    
    # config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    # config.init_device = 'cuda'
    # config.attn_config['attn_impl'] = 'torch'
    # config.attn_config['alibi'] = True
    # config.max_seq_len = 1024

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        # device_map={'': current_device},
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir='./cache',
    )
    model.config.pretraining_tp = 1

    print_gpu_utilization()
    print('Done!')
    # Define training args
    print('Training...')
    output_dir = args.checkpoint_local_path
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        bf16=args.bf16,  # Use BF16 if available
        # tf32=True,
        # learning rate
        lr_scheduler_type='linear',
        learning_rate=args.lr,
        warmup_steps=20,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=3000,
        save_total_limit=5,
        # ddp_find_unused_parameters=False,
    )
    
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    result = trainer.train()
    print_summary(result)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    save_final = f'{output_dir}/final'
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, 
                                              trust_remote_code=True, 
                                              token=args.hf_token,)
    
    tokenizer.save_pretrained(save_final)
    trainer.save_model(save_final)


def main():
    args = parse_args()
    training_function(args)


if __name__ == "__main__":
    main()
