import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint

# ================= CONFIG =================
MODEL_NAME = "microsoft/trocr-base-printed" 
CROPS_DIR = "train_data/crops"
METADATA_FILE = "train_data/metadata.csv"
OUTPUT_DIR = "./trocr_finetuned_full"

# --- GPU OPTIMIZATION SETTINGS ---
# Increase this to fill GPU memory (Try 32, 48, or 64 for high-end GPUs)
BATCH_SIZE = 16 
# Number of CPU cores to use for data loading (keeps GPU fed)
NUM_WORKERS = 4   

EPOCHS = 15
LEARNING_RATE = 4e-5
# ==========================================

class COCOTextDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx]['file_name']
        text = str(self.df.iloc[idx]['text'])
        
        image_path = os.path.join(self.root_dir, file_name)
        # Open image and ensure RGB
        image = Image.open(image_path).convert("RGB")
        
        # Processor handles resizing and normalizing
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Tokenize text labels
        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.max_target_length
        ).input_ids
        
        # Set padding tokens to -100 so they are ignored in loss calculation
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

def train():
    print(f" Loading Pretrained Model: {MODEL_NAME}...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # Essential model configuration
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # Load Data
    df = pd.read_csv(METADATA_FILE)
    df = df.dropna()
    
    print(f" Training on FULL dataset: {len(df)} samples")
    train_dataset = COCOTextDataset(CROPS_DIR, df, processor)

    # --- Training Arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        fp16=True,                  # Mixed precision (Major speedup for modern GPUs)
        predict_with_generate=True,
        
        # No evaluation, just training
        evaluation_strategy="no",
        
        # SAVE STRATEGY: Save every 500 steps so you can resume often
        save_strategy="steps",
        save_steps=500,             
        save_total_limit=2,         # Keep only the last 2 checkpoints to save disk space
        
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        
        dataloader_num_workers=NUM_WORKERS, # Multiprocessing for data loading
        load_best_model_at_end=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    # --- RESUME LOGIC ---
    # Check if a checkpoint exists in the output directory
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    
    if last_checkpoint:
        print(f" Found checkpoint: {last_checkpoint}. Resuming training from there!")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print(" Starting fresh training...")
        trainer.train()
    
    print(" Saving Final Model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f" Done! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()