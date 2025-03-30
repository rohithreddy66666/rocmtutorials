# ROCm Deep Learning Tutorial

This repository contains a comprehensive tutorial for using AMD GPUs with ROCm for deep learning in PyTorch. The tutorial is designed as a Jupyter notebook that guides you through understanding and benchmarking the performance benefits of ROCm acceleration.

## Prerequisites

- An AMD GPU supported by ROCm (tested with Radeon RX 7900 XT)
- Linux operating system with ROCm drivers installed
- Python 3.9+ environment

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/rohithreddy66666/rocmtutorials.git
cd rocmtutorials

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
pip install numpy==1.26.4  # Use NumPy 1.x for better compatibility
pip install matplotlib pandas scikit-learn jupyter
```

## Tutorial Content

The notebook covers the following topics:

1. **Environment Setup and Verification**
   - Checking PyTorch installation with ROCm support
   - Verifying GPU detection and specifications

2. **Matrix Multiplication Benchmark**
   - Comparing CPU vs GPU performance for matrix operations
   - Visualizing performance differences with increasing matrix sizes
   - Measuring speedup factors for various workloads

3. **CNN Training on CIFAR-10**
   - Training a convolutional neural network on a real dataset
   - Comparing training times on CPU vs GPU
   - Visualizing training metrics and loss curves

4. **GPU Memory Management**
   - Understanding ROCm memory allocation and deallocation
   - Best practices for memory management
   - Visualizing memory usage patterns

5. **Mixed Precision Training**
   - Using FP16/FP32 mixed precision for faster training
   - Measuring performance impact and memory savings
   - Understanding when mixed precision is beneficial

## Key Findings

- ROCm provides significant speedups for matrix operations (typically 10-50x faster for large matrices)
- Deep learning training is substantially accelerated (often 5-20x faster)
- Mixed precision training behavior varies by GPU model and workload size
- Proper memory management techniques are essential for optimal performance

## Running the Tutorial

Start Jupyter notebook and open the tutorial:

```bash
jupyter notebook ROCm_Deep_Learning_Tutorial.ipynb
```

Execute each cell sequentially to understand the concepts and see the performance differences between CPU and ROCm GPU acceleration.

## Troubleshooting

- **NumPy Version Issues**: If you encounter NumPy errors, ensure you're using version 1.26.4 or earlier
- **Memory Errors**: If you encounter out-of-memory errors, reduce batch sizes or model sizes
- **ROCm Not Detected**: Verify your ROCm installation with `rocminfo` and ensure PyTorch was installed with ROCm support

## Additional Resources

- [ROCm Documentation](https://rocmdocs.amd.com/en/latest/)
- [PyTorch ROCm Guide](https://pytorch.org/docs/stable/notes/hip.html)
- [AMD GPU Developer Resources](https://developer.amd.com/resources/rocm-resources/)

## License

This tutorial is provided under the MIT License. Feel free to use, modify, and share it according to the license terms.

## Repository

The official repository for this tutorial is: https://github.com/rohithreddy66666/rocmtutorials