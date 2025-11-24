import logging
import pandas as pd
from sentence_transformers import (
    SentenceTransformer, 
    SentenceTransformerTrainer, 
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.util import cos_sim
from torch import nn, Tensor
import torch
from typing import Iterable, Dict
from datasets import Dataset
from peft import LoraConfig, TaskType, PeftModel
from scipy.stats import spearmanr
from tqdm.autonotebook import tqdm
import os

# Set up logger for cleaner output
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# =================================================================================
# 1. Custom Loss Function (Unchanged)
# =================================================================================
class ContrastiveLossWithHardNegatives(nn.Module):
    def __init__(self, model: SentenceTransformer, temperature: float = 0.05, hard_negative_weight: float = 1.0):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # Encode to embeddings
        embeddings = [self.model(feature)['sentence_embedding'] for feature in sentence_features]
        anchor_embeddings, positive_embeddings, hard_neg_embeddings = embeddings
        
        batch_size = anchor_embeddings.size(0)
        device = anchor_embeddings.device

        scores_pos = cos_sim(anchor_embeddings, positive_embeddings) / self.temperature
        scores_hard_neg = cos_sim(anchor_embeddings, hard_neg_embeddings) / self.temperature
        
        scores = torch.cat([scores_pos, scores_hard_neg], dim=1)
        
        if self.hard_negative_weight != 0:
            hard_neg_mask = torch.zeros_like(scores)
            hard_neg_indices = torch.arange(batch_size, device=device)
            hard_neg_mask[hard_neg_indices, batch_size + hard_neg_indices] = self.hard_negative_weight
            scores += hard_neg_mask
            
        labels = torch.arange(batch_size, device=device)
        return self.loss_fct(scores, labels)

# =================================================================================
# 2. Robust Custom Evaluator (Added from reference)
# =================================================================================
class RobustEmbeddingSimilarityEvaluator(EmbeddingSimilarityEvaluator):
    """
    This evaluator is designed to be robust against models that produce
    incorrectly shaped embeddings when encoding a batch of sentences.
    
    It works by encoding each sentence individually and then correctly calculates 
    the final STS Spearman score (correlation * 100) by using the diagonal
    of the similarity matrix.
    """
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            out_txt = f" after epoch {epoch}:" if steps == -1 else f" in epoch {epoch} after {steps} steps:"
            logging.info(f"EmbeddingSimilarityEvaluator: Evaluating the model on the {self.name} dataset{out_txt}")

        logging.info(f"Encoding {len(self.sentences1)} sentences for column 1 (one by one)...")
        embeddings1_list = [
            model.encode(s, convert_to_tensor=True, show_progress_bar=False) 
            for s in tqdm(self.sentences1, desc="Encoding Sentences 1")
        ]
        
        logging.info(f"Encoding {len(self.sentences2)} sentences for column 2 (one by one)...")
        embeddings2_list = [
            model.encode(s, convert_to_tensor=True, show_progress_bar=False) 
            for s in tqdm(self.sentences2, desc="Encoding Sentences 2")
        ]

        if not embeddings1_list or not embeddings2_list:
             logging.error("Encoding resulted in empty lists. Cannot proceed with evaluation.")
             return 0.0

        embeddings1 = torch.stack(embeddings1_list)
        embeddings2 = torch.stack(embeddings2_list)
        
        cosine_scores_matrix = cos_sim(embeddings1, embeddings2)
        paired_cosine_scores = cosine_scores_matrix.diag()

        spearman_result = spearmanr(self.scores, paired_cosine_scores.cpu().numpy())
        sts_spearman_score = spearman_result.correlation * 100.0

        logging.info(f"Spearman score on {self.name}: {sts_spearman_score:.2f}")

        return sts_spearman_score


def enable_dropout_for_simcse(model: torch.nn.Module, dropout_rate: float = 0.1):
    """
    Correctly enables dropout for SimCSE on modern transformer models like Qwen2.
    This function is critical because Qwen2 uses functional dropout controlled by a
    config attribute (`attention_dropout`) rather than a separate nn.Dropout layer.
    """
    logging.info("\n" + "="*80)
    logging.info(f"Activating dropout for SimCSE with rate: {dropout_rate}")
    logging.info("="*80)

    # Set the model to training mode to ensure dropout is active
    model.train()

    # --- Primary Fix: Modify the model's configuration ---
    if hasattr(model.config, 'attention_dropout'):
        logging.info(f"Found 'attention_dropout' in config. Current value: {model.config.attention_dropout}")
        model.config.attention_dropout = dropout_rate
        logging.info(f"✓ Set model.config.attention_dropout to: {model.config.attention_dropout}")
    else:
        logging.warning("⚠️ 'attention_dropout' not found in model config.")

    # --- Secondary Fix: Iterate over modules for explicit nn.Dropout layers (e.g., LoRA) ---
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            if module.p != dropout_rate:
                logging.info(f"Found nn.Dropout module: '{name}'. Current rate: {module.p}")
                module.p = dropout_rate
                logging.info(f"✓ Set module '{name}' dropout rate to: {module.p}")
                
    logging.info("🎉 Dropout configuration complete.")
    logging.info("="*80 + "\n")


def main():
    # =================================================================================
    # 3. CONFIGURATION "CONTROL PANEL" (UPDATED)
    # =================================================================================
    config = {
        # CRITICAL: Specify the ORIGINAL base model path
        "base_model_name_or_path": "Alibaba-NLP/gte-Qwen2-7B-instruct",
        
        # Path to the LoRA checkpoint you want to continue training from
        # Set to None if starting fresh LoRA training
        "lora_checkpoint_path": "/workspace/SDJC_with_instruction_prompt/output_pretrain_wiki_corpus/checkpoint-50",
        
        "prefix": None,
        "instruction_template": "Instruct: Retrieve semantically similar text.\nQuery: {text}",
        #"instruction_template": None,
        "prompt_name": None,
        "trust_remote_code": True,
        "normalize_embeddings": True,
        "max_seq_length": 128,

        # Common settings for all configs below
        "epochs": 5,
        "batch_size": 4,
        "gradient_accumulation_steps": 32,
        "temperature": 0.05,
        "hard_negative_weight": 0.1,

        # Optimizer settings
        "learning_rate": 5e-5,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "scheduler_type": "cosine",

        # Checkpoint and evaluation settings
        "checkpoint_save_steps": 5,
        #"eval_steps": 5,
        "checkpoint_save_total_limit": 5,

        # Data paths
        "train_data_path": "/workspace/SimCSE:v1/data/qa_shuffle_for_simcse_top6.csv",
        #"eval_data_path": "",
        
        # Training mode: "lora" or "full"
        "training_mode": "lora",
        
        # LoRA configuration (only used if training_mode == "lora")
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    
    # Define output paths based on the new API structure
    output_dir = './output_edu_finetune_after_pretrain_batchsize128_lr5e-5/checkpoints'
    final_model_path = './output_edu_finetune_after_pretrain_batchsize128_lr5e-5/final_model'

    # =================================================================================
    # 4. Load Base Model and Apply LoRA Adapter (FIXED)
    # =================================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # STEP 1: Load the base model first
    logging.info(f"Loading base model '{config['base_model_name_or_path']}' on device '{device}'")
    model = SentenceTransformer(
        config["base_model_name_or_path"], 
        device=device, 
        trust_remote_code=config.get("trust_remote_code", False)
    )
    
    # STEP 2: Apply LoRA adapter
    training_mode = config.get("training_mode", "lora")
    
    if training_mode == "lora":
        lora_checkpoint_path = config.get("lora_checkpoint_path")
        
        if lora_checkpoint_path and os.path.exists(lora_checkpoint_path):
            # Load existing LoRA adapter from checkpoint
            logging.info(f"Loading existing LoRA adapter from: {lora_checkpoint_path}")
            try:
                model[0].auto_model = PeftModel.from_pretrained(
                    model[0].auto_model,
                    lora_checkpoint_path,
                    is_trainable=True  # Important: make it trainable
                )
                logging.info("✓ Successfully loaded LoRA adapter from checkpoint")
            except Exception as e:
                logging.error(f"Failed to load LoRA adapter: {e}")
                logging.info("Creating new LoRA adapter instead...")
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    inference_mode=False,
                    r=config.get("lora_r", 32),
                    lora_alpha=config.get("lora_alpha", 64),
                    lora_dropout=config.get("lora_dropout", 0.1),
                    target_modules=config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                )
                model.add_adapter(peft_config)
        else:
            # Create new LoRA adapter
            logging.info("No checkpoint path provided or path doesn't exist. Creating new LoRA adapter...")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=config.get("lora_r", 32),
                lora_alpha=config.get("lora_alpha", 64),
                lora_dropout=config.get("lora_dropout", 0.1),
                target_modules=config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            )
            model.add_adapter(peft_config)
        
        # Set to training mode and ensure gradients are enabled
        model[0].auto_model.train()
        for name, param in model[0].auto_model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
                
        logging.info("LoRA configuration complete")
    else:
        logging.info("Configuring full fine-tuning mode...")
        # Ensure all parameters require gradients
        for param in model.parameters():
            param.requires_grad = True
        logging.info("All model parameters set to require gradients")
    
    # Verify that at least some parameters require gradients
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found! Training cannot proceed.")
    
    # Print LoRA-specific info if in LoRA mode
    if training_mode == "lora":
        lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n.lower() and p.requires_grad)
        logging.info(f"LoRA parameters: {lora_params:,} ({100 * lora_params / total_params:.4f}% of total)")
    
    print(model[0].auto_model)

    # Call the new, corrected function to enable dropout properly.
    enable_dropout_for_simcse(model[0].auto_model, dropout_rate=0.1)

    # Your original diagnostic check remains here to verify the fix.
    logging.info("\n" + "="*50)
    logging.info("RUNNING FINAL DROPOUT DIAGNOSTIC")
    logging.info("="*50)

    model[0].auto_model.train()  # Ensure model is in training mode

    dropout_found = False
    for name, module in model[0].auto_model.named_modules():
        if isinstance(module, nn.Dropout):
            dropout_found = True
            logging.info(f"✓ {name}")
            logging.info(f"  - Dropout rate: {module.p}")
            logging.info(f"  - Training mode: {module.training}")

    if not dropout_found:
        logging.warning("⚠️ WARNING: NO EXPLICIT nn.Dropout LAYERS FOUND!")

    # The most important check for Qwen2:
    if hasattr(model[0].auto_model.config, 'attention_dropout'):
        logging.info(f"\n✓ Config attention_dropout: {model[0].auto_model.config.attention_dropout}")
        if model[0].auto_model.config.attention_dropout == 0.0:
            logging.error("CRITICAL ERROR: attention_dropout is still 0.0! SimCSE will fail.")
    else:
        logging.error("CRITICAL ERROR: attention_dropout attribute not found! SimCSE will fail.")

    logging.info("="*50 + "\n")    

    if config.get("max_seq_length"):
        model.max_seq_length = config["max_seq_length"]
        logging.info(f"Set max_seq_length to {model.max_seq_length}")
    else:
        logging.info(f"Using model's default max_seq_length: {model.max_seq_length}")

    if config.get("prompt_name"):
        model.prompt_name = config["prompt_name"]
        logging.info(f"Set model's default prompt_name to: '{model.prompt_name}'")

    if config.get("normalize_embeddings"):
        model.normalize_embeddings = config["normalize_embeddings"]
        logging.info("Enabled embedding normalization.")

    # =================================================================================
    # 5. Load Data and Apply Manual Prompting Logic (Same as before)
    # =================================================================================
    instruction_template = config.get("instruction_template")
    prefix = config.get("prefix")

    if instruction_template:
        logging.info(f"Applying instruction template to all texts: '{instruction_template}'")
    elif prefix:
        logging.info(f"Applying prefix to all texts: '{prefix}'")
    else:
        logging.info("No manual prefix or template applied. Using raw text (or model's default prompt_name).")

    # --- Load Training Data ---
    train_file_path = config.get('train_data_path')
    logging.info(f"Loading training data from: {train_file_path}")
    df_train = pd.read_csv(train_file_path)
    df_train = df_train.dropna(subset=['sent0', 'sent1', 'hard_neg'])
    logging.info(f"Loaded {len(df_train)} training samples after dropping missing values.")
    
    data = {'anchor': [], 'positive': [], 'negative': []}
    for _, row in df_train.iterrows():
        texts = [row['sent0'], row['sent1'], row['hard_neg']]
        
        if instruction_template:
            formatted_texts = [instruction_template.format(text=t) for t in texts]
        elif prefix:
            formatted_texts = [f"{prefix}{t}" for t in texts]
        else:
            formatted_texts = texts
        
        data['anchor'].append(formatted_texts[0])
        data['positive'].append(formatted_texts[1])
        data['negative'].append(formatted_texts[2])

    train_dataset = Dataset.from_dict(data)
    logging.info(f"Converted training data to datasets.Dataset with {len(train_dataset)} samples.")

    """
    # --- Load Evaluation Data ---
    eval_file_path = config.get('eval_data_path')
    logging.info(f"Loading Japanese evaluation data from: {eval_file_path}")
    dev_samples = []
    df_dev = pd.read_csv(eval_file_path, sep='\t', header=None, names=['score', 'sentence1', 'sentence2'], encoding='utf-8')
    df_dev.dropna(subset=['score', 'sentence1', 'sentence2'], inplace=True)
    df_dev['score'] = pd.to_numeric(df_dev['score']) / 5.0
    
    for _, row in df_dev.iterrows():
        s1, s2 = str(row['sentence1']), str(row['sentence2'])
        if instruction_template:
            s1 = instruction_template.format(text=s1)
            s2 = instruction_template.format(text=s2)
        elif prefix:
            s1 = f"{prefix}{s1}"
            s2 = f"{prefix}{s2}"
            
        dev_samples.append(InputExample(texts=[s1, s2], label=row['score']))

    if not dev_samples:
        raise ValueError(f"Evaluation data loading failed. 0 samples were loaded from '{eval_file_path}'. Check file path, format (should be 3 columns: score, sentence1, sentence2), and encoding.")

    evaluator = RobustEmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='japanese-sts-dev')
    logging.info(f"Loaded {len(dev_samples)} evaluation samples for Japanese STS.")
    """
    # =================================================================================
    # 6. Define Loss and Train the Model
    # =================================================================================
    train_loss = ContrastiveLossWithHardNegatives(
        model=model,
        temperature=config['temperature'],
        hard_negative_weight=config['hard_negative_weight']
    )

    effective_batch_size = config['batch_size'] * config['gradient_accumulation_steps']
    logging.info(f"Effective Batch Size: {effective_batch_size}")

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        warmup_ratio=config['warmup_ratio'],
        lr_scheduler_type=config['scheduler_type'],
        optim="adamw_torch",
        adam_beta1=config['adam_beta1'],
        adam_beta2=config['adam_beta2'],
        adam_epsilon=config['adam_epsilon'],
        weight_decay=config['weight_decay'],
        save_strategy="steps",
        save_steps=config['checkpoint_save_steps'],
        #save_total_limit=config.get("checkpoint_save_total_limit"),
        logging_strategy="steps",
        logging_steps=config['checkpoint_save_steps'],
        #eval_strategy="steps",
        #eval_steps=config['eval_steps'],
        #load_best_model_at_end=True,
        #metric_for_best_model="eval_evaluator",
        remove_unused_columns=False,
        bf16=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        #evaluator=evaluator,
    )

    logging.info("Starting model training with the SentenceTransformerTrainer...")
    trainer.train()
    
    # =================================================================================
    # 7. Save the Final Model (IMPROVED)
    # =================================================================================
    if training_mode == "lora":
        # Save LoRA adapter only
        adapter_output_path = final_model_path + "_adapter"
        logging.info(f"Saving LoRA adapter to '{adapter_output_path}'")
        model[0].auto_model.save_pretrained(adapter_output_path)
        
        # Also save a merged version for easier deployment
        logging.info("Merging LoRA weights into base model...")
        model[0].auto_model = model[0].auto_model.merge_and_unload()
        logging.info(f"Saving merged model to '{final_model_path}'")
        model.save(final_model_path)
        
        logging.info(f"Training complete!")
        logging.info(f"  - LoRA adapter saved to: {adapter_output_path}")
        logging.info(f"  - Merged model saved to: {final_model_path}")
    else:
        # Save full model
        trainer.save_model(final_model_path)
        logging.info(f"Training complete. Model saved to '{final_model_path}'.")


if __name__ == "__main__":
    main()