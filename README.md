# ROCm Tutorials Repository

Welcome to the ROCm Tutorials Repository! This collection of tutorials is designed to help developers harness the power of AMD GPUs using the ROCm (Radeon Open Compute) platform for high-performance computing and deep learning.

## About This Repository

This repository contains a growing collection of practical, hands-on tutorials that demonstrate how to leverage ROCm for various computing tasks. Each tutorial is contained in its own folder with complete code examples and documentation.

## Available Tutorials

### 1. [CPU vs GPU Performance Comparison](./cpu_vs_gpu/)

A comprehensive tutorial comparing CPU and GPU performance for common deep learning tasks. This tutorial demonstrates:

- Matrix multiplication performance benchmarks
- CNN training with CIFAR-10 dataset
- GPU memory management techniques
- Mixed precision training comparisons

Ideal for: Anyone new to ROCm who wants to understand the performance benefits of AMD GPUs for deep learning.

### 2. [Finetuning LLMs using ROCm and LoRA](./finetuning_llms_using_rocm_and_lora/)

A tutorial on how to efficiently finetune large language models using AMD GPUs with ROCm and LoRA:
* Setting up the ROCm environment for LLM finetuning
* Implementing Parameter-Efficient Fine-Tuning (PEFT) with LoRA
* Memory optimization techniques for large models
* Performance benchmarks on AMD hardware

Ideal for: ML practitioners looking to customize foundation models on AMD hardware with limited resources.

## Upcoming Tutorials

Stay tuned for more tutorials covering:

- ROCm with PyTorch for large language models
- Multi-GPU training with ROCm
- Advanced optimization techniques for AMD GPUs
- Computer vision applications with ROCm
- Converting CUDA code to run on ROCm

## Requirements

- AMD GPU with ROCm support
- Linux operating system with ROCm drivers installed
- Python 3.9+ with PyTorch
- Additional requirements may vary per tutorial

## Getting Started

Clone this repository to get started:

```bash
git clone https://github.com/rohithreddy66666/rocmtutorials.git
cd rocmtutorials
```

Then navigate to any tutorial directory and follow the instructions in its README file.

## Contributing

Contributions to this repository are welcome! If you'd like to contribute a tutorial or improve existing ones, please:

1. Fork the repository
2. Create a new branch for your changes
3. Submit a pull request with a clear description of your contribution

## License

This repository is provided under the MIT License. See individual tutorials for any specific licensing information.

## Resources

- [ROCm Documentation](https://rocmdocs.amd.com/en/latest/)
- [PyTorch ROCm Guide](https://pytorch.org/docs/stable/notes/hip.html)
- [AMD GPU Developer Resources](https://developer.amd.com/resources/rocm-resources/)

## Contact

For questions or feedback about these tutorials, please open an issue in this repository.
