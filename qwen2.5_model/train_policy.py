#!/usr/bin/env python3
"""Training script for fine-tuning Qwen2.5-1.5B on Rego policy rules.

This script is specialized for policy rule training with:
- Conservative training settings for small datasets (~260 examples)
- LoRA configuration targeting attention + MLP layers
- Support for implement and refactor task types
- Focused context (package + imports + helpers only)
"""

# Set tokenizers parallelism to avoid warnings when using dataloader workers
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix CUDA/NVML initialization issues
# Set CUDA device before importing torch to avoid NVML errors
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU by default

# Disable CUDA memory pool issues
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import argparse
import json
import sys
import logging
import torch

# Initialize CUDA properly before any operations
# This helps avoid NVML initialization errors
# Note: torch.cuda.init() doesn't exist - CUDA initializes automatically
# We just need to ensure proper device selection
try:
    if torch.cuda.is_available():
        # Access CUDA device to trigger initialization
        _ = torch.cuda.current_device()
        torch.cuda.empty_cache()
except Exception as e:
    # Will log after logger is set up
    pass  # Logged later
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from torch.utils.data import DataLoader

# Import logging setup
try:
    from logging_setup import setup_logging, log_exception
except ImportError:
    # Fallback if logging_setup not available
    import logging
    logging.basicConfig(level=logging.INFO)
    def setup_logging(name, **kwargs):
        return logging.getLogger(name)
    def log_exception(logger, exc, context=""):
        logger.error(f"{context}: {exc}" if context else f"Exception: {exc}", exc_info=True)

# PEFT is optional - support full fine-tuning if not available
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    # Will log this after logger is set up


@dataclass
class TrainingConfig:
    """Configuration for policy rule training."""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    train_path: str = "qwen2.5_model/train.jsonl"
    eval_path: str = "qwen2.5_model/eval.jsonl"
    output_dir: str = "qwen2.5-rego-policy-lora"
    max_seq_len: int = 1024  # Conservative for small examples (avg ~300 tokens)
    batch_size: int = 2  # Conservative for MPS memory constraints
    grad_accum_steps: int = 4  # Effective batch size = 8
    learning_rate: float = 5e-5  # Conservative for small datasets
    num_epochs: int = 3  # Max 3 for small datasets
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    warmup_steps: int = 50  # Fewer warmup steps for small dataset
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100


QWEN_SYSTEM_PROMPT_POLICY = (
    "You are an expert Rego/OPA policy assistant. "
    "You follow instructions carefully and emit valid Rego code using "
    "Conforma's preferred patterns (deny contains result, METADATA, result_helper, etc). "
    "Only use helpers that are provided in the context - never invent new helper functions."
)

QWEN_SYSTEM_PROMPT_ATTESTATION = (
    "You are an expert Rego policy rule writer. Rego is a declarative policy language for evaluating structured data like JSON. "
    "Given an instruction about what to find or check in an attestation, write Rego code that evaluates the attestation JSON structure and makes the requested policy decision. "
    "Use proper Rego syntax: declarative expressions, array iteration with 'some', field access, and condition checking. "
    "Express \"what should be true\" rather than \"how to check it\"."
)


def build_messages_from_example(example):
    """Build chat messages from policy training example.
    
    Format:
    - instruction: Task description
    - context: Package + imports + helpers (for policy rules) or JSON attestation (for attestation parsing)
    - input_code: (optional) Code to refactor
    - output_code: Expected output code
    - task_type: "implement", "refactor", or "rego_attestation_parse"
    """
    # Select system prompt based on task type
    task_type = example.get("task_type", "implement")
    if task_type == "rego_attestation_parse":
        system_prompt = QWEN_SYSTEM_PROMPT_ATTESTATION
    else:
        system_prompt = QWEN_SYSTEM_PROMPT_POLICY
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Build user message
    user_parts = []
    
    # Add context (package + imports + helpers)
    if "context" in example:
        user_parts.append(example["context"])
    
    # Add instruction
    if "instruction" in example:
        user_parts.append("\n" + example["instruction"])
    
    # For refactor tasks, include input_code
    if example.get("task_type") == "refactor" and "input_code" in example:
        user_parts.append("\n\nCode to refactor:\n```rego\n" + example["input_code"] + "\n```")
    
    user_content = "\n".join(user_parts)
    messages.append({"role": "user", "content": user_content})
    
    # Assistant response is the output_code
    if "output_code" in example:
        messages.append({"role": "assistant", "content": example["output_code"]})
    
    return messages


def rego_collate_fn(batch, pad_token_id):
    """Dynamic padding collator - pads each batch to the longest example in that batch."""
    input_ids_list = [b["input_ids"] for b in batch]
    labels_list = [b["labels"] for b in batch]
    
    # Find the longest sequence in this batch
    max_len = max(t.size(0) for t in input_ids_list)
    
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    
    for ids, lbls in zip(input_ids_list, labels_list):
        pad_len = max_len - ids.size(0)
        
        if pad_len > 0:
            pad_ids = torch.full((pad_len,), pad_token_id, dtype=ids.dtype)
            pad_lbls = torch.full((pad_len,), -100, dtype=lbls.dtype)
            ids = torch.cat([ids, pad_ids], dim=0)
            lbls = torch.cat([lbls, pad_lbls], dim=0)
        
        padded_input_ids.append(ids)
        padded_labels.append(lbls)
        attention_masks.append(torch.ones_like(ids, dtype=torch.long))
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(padded_labels),
    }


class PolicyDataset(Dataset):
    """Dataset for policy training examples.
    
    Pre-tokenizes all examples in __init__ to avoid repeated tokenization,
    which dramatically speeds up training.
    """
    
    def __init__(self, jsonl_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        self.input_ids = []
        self.labels = []
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load samples
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        
        # Pre-tokenize everything once
        logging.info(f"Pre-tokenizing {len(self.samples)} examples...")
        for i, example in enumerate(self.samples):
            if (i + 1) % 50 == 0:
                logging.info(f"  Tokenized {i + 1}/{len(self.samples)} examples...")
            
            # Build messages
            messages = build_messages_from_example(example)
            
            # Tokenize full conversation (system + user + assistant)
            full_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            ).squeeze(0)
            
            # Tokenize just the prompt (system + user) to find where assistant starts
            prompt_messages = messages[:2]  # system + user
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).squeeze(0)
            
            # Build labels (mask prompt)
            labels = full_ids.clone()
            labels[:len(prompt_ids)] = -100
            
            # Truncate from the left if over max_seq_len (preserve assistant response)
            if full_ids.size(0) > self.max_seq_len:
                full_ids = full_ids[-self.max_seq_len:]
                labels = labels[-self.max_seq_len:]
            
            # Store pre-computed tensors (without padding)
            self.input_ids.append(full_ids)
            self.labels.append(labels)
        
        logging.info("Pre-tokenization complete")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Return pre-computed tensors - no tokenization needed!"""
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }


def load_qwen_model(cfg, enable_gradient_checkpointing=True):
    """Load Qwen model and tokenizer for MPS/CUDA/CPU."""
    # Detect available device
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple GPU
        device_name = "MPS (Apple Silicon GPU)"
    elif torch.cuda.is_available():
        try:
            # Try to initialize CUDA properly
            torch.cuda.init()
            device = torch.device("cuda")
            device_name = f"CUDA (GPU: {torch.cuda.get_device_name(0)})"
        except Exception as e:
            logging.warning(f"CUDA initialization failed: {e}")
            logging.warning("  Falling back to CPU")
            device = torch.device("cpu")
            device_name = "CPU"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    
    logging.info(f"Using device: {device_name}")
    dtype = torch.bfloat16  # Qwen prefers bf16/fp16

    logging.info(f"Loading tokenizer from {cfg.model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
        )
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        logging.error("Possible solutions:")
        logging.error("1. Check if the model name is correct on HuggingFace")
        logging.error("2. If it's a gated model, authenticate with: huggingface-cli login")
        logging.error("3. Try: Qwen/Qwen2.5-1.5B-Instruct")
        log_exception(logging.getLogger(__name__), e, "Failed to load tokenizer")
        raise
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    logging.info(f"Loading model from {cfg.model_name}...")
    try:
        device_map_value = str(device) if device.type != "cpu" else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            device_map={"": device_map_value} if device.type != "cpu" else None,
            trust_remote_code=True,
        )
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.error("\nPossible solutions:")
        logging.error("1. Check if the model name is correct on HuggingFace")
        logging.error("2. If it's a gated model, authenticate with: huggingface-cli login")
        logging.error("3. Try a smaller model if you're running out of memory")
        log_exception(logging.getLogger(__name__), e, "Failed to load model")
        raise

    # Always disable cache for training
    if hasattr(model, "config"):
        model.config.use_cache = False

    if enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # Move model to device (if not already there via device_map)
    if device.type == "cpu" or not hasattr(model, "hf_device_map"):
        model.to(device)
    model.train()
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    model_device = next(model.parameters()).device
    logging.info(f"Loaded model with {param_count:.2f}B parameters")
    logging.info(f"Model on device: {model_device}")
    return tokenizer, model


def apply_lora(model, cfg, use_lora=True):
    """Apply LoRA to the model, or return model as-is for full fine-tuning."""
    if not use_lora or not PEFT_AVAILABLE:
        logging.info("Using full fine-tuning (all parameters trainable)")
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logging.info(f"  Total parameters: {total_params:,}")
        return model
    
    logging.info("Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            # Attention layers
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            # MLP layers
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )

    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model


def train_with_trainer(model, tokenizer, train_dataset, eval_dataset, cfg, 
                       disable_gradient_checkpointing=False, 
                       dataloader_num_workers=0, eval_strategy="steps"):
    """Train the model using HuggingFace Trainer."""
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        
        # Precision settings
        bf16=False,  # Disable autocast - model dtype (bfloat16) already set
        fp16=False,  # Don't use fp16 on MPS
        
        # Memory optimization
        gradient_checkpointing=not disable_gradient_checkpointing,
        # Pin memory: True for CUDA (faster), False for MPS (required)
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_num_workers=dataloader_num_workers,
        
        # Performance optimizations
        dataloader_drop_last=True,
        max_grad_norm=1.0,
        
        # MPS-specific optimizations
        dataloader_prefetch_factor=None,  # Let PyTorch decide for MPS
        
        # Optimizer
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        
        # Logging and evaluation
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        eval_strategy=eval_strategy,
        
        # MPS-specific: Skip first evaluation to avoid hang
        skip_memory_metrics=False,  # Keep memory metrics for debugging
        
        # Checkpointing
        save_total_limit=3,
        load_best_model_at_end=(eval_strategy != "no"),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Learning rate schedule
        lr_scheduler_type="cosine",
        warmup_steps=cfg.warmup_steps,
        
        # Other settings
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Create dynamic padding collator
    data_collator = lambda batch: rego_collate_fn(batch, tokenizer.pad_token_id)
    
    # For MPS, warm up with a small batch first to avoid hangs
    if torch.backends.mps.is_available() and eval_strategy != "no":
        logging.info("MPS detected with evaluation enabled - warming up model...")
        # Create a dummy batch to warm up MPS
        dummy_batch = data_collator([train_dataset[0], train_dataset[1]])
        with torch.no_grad():
            dummy_output = model(**{k: v.to(model.device) for k, v in dummy_batch.items() if k != "labels"})
        torch.mps.synchronize()
        logging.info("Model warmed up")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    logging.info("Starting training...")
    logging.warning("Monitor eval loss closely - stop early if it starts increasing (sign of overfitting)")
    
    # For MPS, add explicit synchronization before training starts
    if torch.backends.mps.is_available():
        logging.debug("MPS detected - adding synchronization point...")
        torch.mps.synchronize()
    
    # Add progress callback to debug hangs
    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.step_count = 0
        
        def on_step_end(self, args, state, control, **kwargs):
            self.step_count += 1
            if self.step_count % 10 == 0:
                logging.info(f"  Progress: {self.step_count} steps completed")
            return control
    
    trainer.add_callback(ProgressCallback())
    
    trainer.train()
    return trainer


def save_lora_model(model, tokenizer, output_dir):
    """Save LoRA adapters and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving LoRA adapters to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Saved LoRA adapters and tokenizer to {output_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Qwen2.5-1.5B on Rego policy rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (conservative for small datasets)
  uv run python qwen2.5_model/train_policy.py \\
      --train-path qwen2.5_model/train.jsonl \\
      --eval-path qwen2.5_model/eval.jsonl \\
      --output-dir qwen2.5-rego-policy-lora

  # Full fine-tuning (without LoRA/PEFT)
  uv run python qwen2.5_model/train_policy.py \\
      --train-path qwen2.5_model/attestation_train.jsonl \\
      --eval-path qwen2.5_model/attestation_eval.jsonl \\
      --output-dir qwen2.5-attestation-parse \\
      --no-lora

  # For faster training (if you have enough memory)
  uv run python qwen2.5_model/train_policy.py \\
      --train-path qwen2.5_model/train.jsonl \\
      --eval-path qwen2.5_model/eval.jsonl \\
      --output-dir qwen2.5-rego-policy-lora \\
      --disable-gradient-checkpointing \\
      --batch-size 4
        """
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        required=True,
        help="Path to evaluation JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="qwen2.5-rego-policy-lora",
        help="Output directory for LoRA adapters",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Maximum sequence length for truncation (default: 1024, covers most examples)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per device (default: 2 for MPS memory safety)",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4, effective batch size = batch_size * grad_accum_steps)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5 for small datasets)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3 max for small datasets)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Warmup steps for cosine decay",
    )
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (MUCH faster but uses more memory)",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading (default: 0)",
    )
    parser.add_argument(
        "--eval-strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy (default: 'steps')",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA and use full fine-tuning (required if PEFT not available)",
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to repository root
    script_dir = Path(__file__).parent.resolve()
    current_dir = Path.cwd()
    
    # Find repo root
    repo_root = None
    for candidate in [script_dir, current_dir]:
        check_dir = candidate
        for _ in range(5):
            if (check_dir / ".git").exists() or (check_dir / "policy").exists():
                repo_root = check_dir
                break
            check_dir = check_dir.parent
        if repo_root:
            break
    
    if not repo_root:
        repo_root = current_dir
    
    def resolve_path(path: str) -> str:
        """Resolve path relative to repo root if it's relative."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        if path_obj.exists():
            return str(path_obj.resolve())
        repo_path = repo_root / path
        if repo_path.exists():
            return str(repo_path.resolve())
        return str(path_obj)
    
    train_path = resolve_path(args.train_path)
    eval_path = resolve_path(args.eval_path)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else str(repo_root / args.output_dir)
    
    # Create config from args
    cfg = TrainingConfig(
        model_name=args.model_name,
        train_path=train_path,
        eval_path=eval_path,
        output_dir=output_dir,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        warmup_steps=args.warmup_steps,
    )
    
    # Validate paths
    if not os.path.exists(cfg.train_path):
        raise FileNotFoundError(f"Training file not found: {cfg.train_path}")
    if not os.path.exists(cfg.eval_path):
        raise FileNotFoundError(f"Evaluation file not found: {cfg.eval_path}")
    
    # Setup logging
    logger = setup_logging("train_policy")
    
    logger.info("=" * 60)
    logger.info("Policy Rule Model Training")
    logger.info("=" * 60)
    
    # Log PEFT availability
    if not PEFT_AVAILABLE:
        logger.warning("PEFT not available - will use full fine-tuning instead of LoRA")
    
    # Load model and tokenizer
    enable_model_gc = not args.disable_gradient_checkpointing
    tokenizer, base_model = load_qwen_model(cfg, enable_gradient_checkpointing=enable_model_gc)
    
    # Determine if we should use LoRA
    use_lora = not args.no_lora and PEFT_AVAILABLE
    if args.no_lora:
        logger.info("--no-lora flag set: using full fine-tuning")
    elif not PEFT_AVAILABLE:
        logger.info("PEFT not available: using full fine-tuning")
    
    model = apply_lora(base_model, cfg, use_lora=use_lora)
    
    # Create datasets (pre-tokenization happens here)
    logger.info("Creating datasets...")
    logger.info("  Note: Pre-tokenizing examples for faster training")
    train_ds = PolicyDataset(cfg.train_path, tokenizer, cfg.max_seq_len)
    eval_ds = PolicyDataset(cfg.eval_path, tokenizer, cfg.max_seq_len)
    logger.info(f"  Training examples: {len(train_ds)}")
    logger.info(f"  Evaluation examples: {len(eval_ds)}")
    
    # Train
    trainer = train_with_trainer(
        model, tokenizer, train_ds, eval_ds, cfg,
        disable_gradient_checkpointing=args.disable_gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        eval_strategy=args.eval_strategy
    )
    
    # Save
    save_lora_model(model, tokenizer, cfg.output_dir)
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)
    logger.info(f"LoRA adapters saved to: {cfg.output_dir}")
    logger.info("To use the model at inference time:")
    logger.info(f"  Load the model from: {cfg.output_dir}")


if __name__ == "__main__":
    main()

