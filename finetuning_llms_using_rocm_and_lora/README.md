# Fine-tuning LLaMA 3.1 on AMD MI250X Using Rocm with LoRA

This repository contains a Jupyter notebook and supporting code for fine-tuning Meta's LLaMA 3.1 8B model using Low-Rank Adaptation (LoRA) with quantization on AMD MI250X GPUs. The implementation is optimized for high-performance computing environments with ROCm support.

## 🚀 Features

- Fine-tune LLaMA 3.1 8B with minimal GPU memory requirements through 4-bit quantization
- Parameter-efficient training using LoRA (only 0.17% of parameters are trainable)
- Optimized for AMD MI250X GPUs using ROCm platform
- Complete Jupyter notebook with step-by-step process and explanations
- Includes BitsAndBytes installation specifically configured for MI250X

## 📋 Hardware Requirements

- AMD MI250X GPU(s)
- ROCm-compatible system
- Minimum 32GB GPU memory (with 4-bit quantization)

## 📦 Installation and Setup

### Install Required Libraries

```bash
pip install pandas peft==0.14.0 transformers==4.47.1 trl==0.13.0 accelerate==1.2.1 scipy tensorboardX
```

### BitsAndBytes Setup for AMD MI250X

BitsAndBytes requires a specific setup for AMD MI250X GPUs. The notebook includes a complete installation process:

```bash
# Remove existing directory, clone and install bitsandbytes specifically for MI250X
rm -rf bitsandbytes && \
git clone --recurse https://github.com/ROCm/bitsandbytes.git && \
cd bitsandbytes && \
git checkout rocm_enabled_multi_backend && \
pip install -r requirements-dev.txt && \
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx90a" -S . && \
make && \
pip install . && \
cd .. && \
python -c "import bitsandbytes as bnb; print('bitsandbytes version:', bnb.__version__)"
```

**Note:** Version 0.43 is the compatible version for the MI250X.

### HuggingFace Authentication

You'll need to authenticate with HuggingFace to access the LLaMA 3.1 model:

```python
from huggingface_hub import login
login(token="your_huggingface_token", add_to_git_credential=False)
```

## 🔧 Model Configuration

The model uses 4-bit quantization to reduce memory usage:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)
```

### LoRA Configuration

The fine-tuning uses LoRA with the following parameters:

```python
peft_parameters = LoraConfig(
    lora_alpha=8,           # Scaling parameter
    lora_dropout=0.1,       # Dropout probability
    r=32,                   # Rank of low-rank matrices
    bias="none",            # No bias parameter training
    task_type="CAUSAL_LM"   # For text generation
)
```

This reduces the number of trainable parameters significantly:
```
trainable params: 13,631,488 || all params: 8,043,892,736 || trainable%: 0.1695
```

## 📚 Dataset Format and Customization

The training data is formatted for LLaMA instruction fine-tuning:

```
<s>[INST] {instruction} {input} [/INST] {output}</s>
```

The notebook extracts this data from a text file containing JSON objects with:
- `instruction`: The instruction prompt
- `input`: Additional context
- `output`: The desired response

### Customizing for Your Own Data

The example notebook is trained on data specific to Rohith (the author), but you can customize it for your own use case:

1. Replace the contents of `rohith.txt` with your own JSON-formatted training examples
2. Update the test queries in the testing section to match your data domain
3. Change output directory names if desired

The data format should be maintained as JSON objects, but the content can be any instruction-response pairs relevant to your specific use case.

## 🔄 Training Configuration

The training uses the following parameters:

```python
training_args = TrainingArguments(
    output_dir="./results_rohith_lora",
    num_train_epochs=50,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=4e-5,
    weight_decay=0.001,
    logging_steps=1,
    save_strategy="epoch",
    max_grad_norm=0.3,
    fp16=True,
    logging_dir="./logs",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="tensorboard"
)
```

Key training features:
- 50 training epochs
- Small batch size (1) with gradient accumulation (4) for effective batch size of 4
- Learning rate of 4e-5 with cosine scheduler and 3% warmup
- Mixed precision (FP16) for efficiency
- TensorBoard integration for monitoring training progress

## 🖥️ Model Testing

After training, you can test the model with the provided testing script that:
1. Loads the base model
2. Merges the LoRA weights
3. Creates a text generation pipeline with controlled parameters:
   - Maximum length of 1024 tokens
   - Temperature of 0.1 for more focused outputs
   - Top-p of 0.3 for higher quality
   - Repetition penalty of 1.2

Example test queries included (these are specific to the author's data and should be replaced with queries relevant to your own dataset):
```python
test_queries = [
    "Detail Rohith's GPU-accelerated document processing pipeline",
    "Explain Rohith's technical documentation work at Radian",
    "What programming languages does Rohith know",
    "What is Rohith's educational background?",
    "What certifications does Rohith have"
]
```

## 📓 Notebook Structure

The Jupyter notebook is organized into well-documented sections:
1. **Required Libraries**: Installation of dependencies
2. **BitsAndBytes Setup**: Special configuration for MI250X
3. **HuggingFace Authentication**: Access to LLaMA 3.1
4. **GPU Setup**: Configuring and verifying GPU availability
5. **Model Configuration**: Setting up quantization
6. **Tokenizer and Model Loading**: Loading the base model
7. **Dataset Preparation**: Processing training data
8. **LoRA and Training Configuration**: Setting up efficient fine-tuning
9. **Training and Saving**: The actual training process
10. **Testing**: Evaluating the fine-tuned model

## 🚫 Troubleshooting

If you encounter issues with BitsAndBytes installation, verify:
- You're using the correct ROCm version for your system
- The bnb_rocm_arch parameter matches your GPU (gfx90a for MI250X)
- You have the necessary C++ build tools installed


## 🙏 Acknowledgements

- [Meta](https://ai.meta.com/) for releasing the LLaMA 3.1 model
- [Hugging Face](https://huggingface.co/) for Transformers, PEFT, and TRL libraries
- [AMD](https://www.amd.com/) for ROCm software platform

## 👤 Author

Rohith Reddy Vangala

---

*For questions or issues, please open an issue in this repository.*
