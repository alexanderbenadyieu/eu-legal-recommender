# summarization/src/abstractive/bart_finetuner.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch

from transformers import pipeline

# Model names
bart_model_name = "MikaSie/BART_no_extraction_V2"  # Tier 1
lexlm_bart_model_name = "MikaSie/LexLM_BART_fixed_V1"  # Tier 2
lexlm_longformer_bart_model_name = "MikaSie/LexLM_Longformer_BART_fixed_V1"  # Tier 3 & 4

# Initialize pipelines lazily
bart_pipeline = None
lexlm_bart_pipeline = None
lexlm_longformer_pipeline = None

def get_pipeline_and_max_length(tier: int):
    """Get the appropriate pipeline and max length for the given tier."""
    global bart_pipeline, lexlm_bart_pipeline, lexlm_longformer_pipeline
    
    if tier == 1:
        if bart_pipeline is None:
            print("Loading BART model...")
            bart_pipeline = pipeline('summarization', model=bart_model_name)
        return bart_pipeline, 1024  # BART max length
    elif tier == 2:
        if lexlm_bart_pipeline is None:
            print("Loading LexLM BART model...")
            lexlm_bart_pipeline = pipeline('summarization', model=lexlm_bart_model_name)
        return lexlm_bart_pipeline, 1024  # LexLM BART max length
    else:  # tier 3 or 4
        if lexlm_longformer_pipeline is None:
            print("Loading LexLM Longformer BART model...")
            lexlm_longformer_pipeline = pipeline('summarization', model=lexlm_longformer_bart_model_name)
        return lexlm_longformer_pipeline, 16384  # Longformer max length

class BartFineTuner:
    def __init__(self, model_name="facebook/bart-large-cnn", is_longformer=False):
        self.is_longformer = is_longformer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def fine_tune(self, train_dataset, eval_dataset=None):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            evaluation_strategy="steps" if eval_dataset else "no",
            logging_dir='./logs',
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer.train()

def bart_summarise(text: str, min_length: int, max_length: int, tier: int = 1) -> str:
    """Summarize text using appropriate BART model based on tier.
    
    Args:
        text: Input text to summarize
        min_length: Minimum length of summary in words
        max_length: Maximum length of summary in words
        tier: Document tier (1-4)
    
    Returns:
        str: Generated summary
    """
    # Get appropriate pipeline and max length for tier
    model, max_input_length = get_pipeline_and_max_length(tier)
    
    # Truncate input if needed (approximate word to token ratio of 1:1.3)
    max_words = int(max_input_length / 1.3)
    words = text.split()
    if len(words) > max_words:
        print(f"Truncating input from {len(words)} to {max_words} words")
        text = ' '.join(words[:max_words])
    
    # Generate summary
    summary = model(text, 
                   min_length=min_length,
                   max_length=max_length,
                   length_penalty=2.0,
                   num_beams=4,
                   early_stopping=True)[0]['summary_text']
    
    return summary
    # Convert word lengths to token lengths (approximate)
    min_tokens = min_length * 1.3  # Approximate conversion
    max_tokens = max_length * 1.3
    
    # Tokenize input text
    inputs = bart_tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )
    
    # Generate summary
    with torch.no_grad():
        summary_ids = bart_model.generate(
            inputs["input_ids"],
            num_beams=4,
            min_length=int(min_tokens),
            max_length=int(max_tokens),
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    # Decode summary
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# We no longer need a separate longformer function since it's handled in bart_summarise
    # Convert word lengths to token lengths (approximate)
    min_tokens = min_length * 1.3  # Approximate conversion
    max_tokens = max_length * 1.3
    
    # Tokenize input text
    inputs = longformer_bart_tokenizer(
        text,
        return_tensors="pt",
        max_length=16384,  # Longformer can handle much longer sequences
        truncation=True
    )
    
    # Generate summary
    with torch.no_grad():
        summary_ids = longformer_bart_model.generate(
            inputs["input_ids"],
            num_beams=4,
            min_length=int(min_tokens),
            max_length=int(max_tokens),
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    # Decode summary
    summary = longformer_bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary