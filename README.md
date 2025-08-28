# SimpleMathOCR - Hand2Tex Training Branch

A comprehensive mathematical OCR system that converts handwritten mathematical expressions to LaTeX format, featuring both training and inference capabilities.

## 🚀 Features

- **Dual Architecture**: Support for both basic classification and advanced decoder-based models
- **Multiple Decoder Types**: RNN (LSTM) and Transformer decoders
- **Comprehensive Symbol Mapping**: 45+ mathematical LaTeX symbols
- **Training Pipeline**: Complete training loop with logging and checkpointing
- **Data Loading**: Support for Kaggle datasets and dummy data for testing
- **Easy Inference**: Simple command-line interface for prediction
- **Extensible**: Easy to add new symbols and decoder types

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SimpleMathOCR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏗️ Project Structure

```
SimpleMathOCR/
├── src/
│   ├── model.py          # Enhanced model architecture
│   ├── decoder.py        # RNN and Transformer decoders
│   ├── symbol_map.py     # LaTeX symbol mapping
│   ├── data.py           # Basic data preprocessing
│   ├── data_kaggle.py    # Kaggle dataset handling
│   ├── utils.py          # Utility functions
│   └── main.py           # Configuration loader
├── configs/
│   └── config.yaml       # Comprehensive configuration
├── tests/
│   └── test_pipeline.py  # Unit tests
├── train.py              # Training script
├── inference.py          # Inference script
└── requirements.txt      # Dependencies
```

## 🎯 Usage

### Training

#### Basic Training (Classification)
```bash
# Train with basic model architecture
python train.py
```

#### Advanced Training (Decoder-based)
```bash
# Train with external decoder (configure in config.yaml)
# Set model.use_decoder: true in config.yaml
python train.py
```

#### Training Configuration
Edit `configs/config.yaml` to customize training:

```yaml
# Data configuration
data:
  kaggle_data_dir: "data/kaggle_math"
  num_classes: 45
  input_size: 128

# Model configuration
model:
  use_decoder: true  # Use external decoder
  hidden_dim: 256

# Training configuration
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 10
  device: "cpu"

# Decoder configuration
decoder_type: "rnn"  # Options: "rnn", "transformer"
max_length: 50
num_layers: 2
```

### Inference

#### Single Image
```bash
python inference.py path/to/image.png
```

#### With Custom Model
```bash
python inference.py path/to/image.png --model models/model_epoch_10.pth
```

#### Save Output to File
```bash
python inference.py path/to/image.png --output result.txt
```

#### Batch Processing
```bash
python inference.py --batch image1.png image2.png image3.png --output results/
```

### Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific tests:
```bash
pytest tests/test_pipeline.py::test_symbol_mapper
pytest tests/test_pipeline.py::test_inference_correctness
```

## 🔧 Configuration

### Model Architecture

The system supports two architectures:

1. **Basic Model**: Simple classification model
   - `use_decoder: false`
   - Direct class prediction

2. **Decoder Model**: Advanced sequence generation
   - `use_decoder: true`
   - Encoder + Decoder architecture
   - Supports RNN and Transformer decoders

### Data Sources

1. **Kaggle Dataset**: Real handwritten math symbols
   - Configure `data.kaggle_data_dir` in config.yaml
   - Automatic fallback to dummy data if not available

2. **Dummy Data**: For testing and development
   - Automatically generated
   - Configurable size and classes

### Supported LaTeX Symbols

45+ mathematical symbols including:

- **Basic operators**: `+`, `-`, `=`, `\times`, `\div`
- **Fractions and roots**: `\frac`, `\sqrt`
- **Greek letters**: `\alpha`, `\beta`, `\pi`
- **Calculus**: `\int`, `\sum`
- **Functions**: `\sin`, `\cos`, `\tan`, `\log`
- **Inequalities**: `\leq`, `\geq`
- **Variables**: `x`, `y`, `z`, `n`
- **Digits**: `0-9`
- **Brackets**: `()`, `[]`, `{}`

## 🧪 Testing

### Quick Start with Dummy Data

```bash
# Train with dummy data (no real dataset needed)
python train.py
```

### Full Training with Real Data

1. Download Kaggle dataset to `data/kaggle_math/`
2. Configure paths in `configs/config.yaml`
3. Run training:
```bash
python train.py
```

### Inference Testing

```bash
# Test with sample image
python inference.py data/sample.png

# Test with custom model
python inference.py data/sample.png --model models/best_model.pth
```

## 📊 Model Architecture

### Encoder
- Enhanced convolutional layers for feature extraction
- Support for different input sizes (64x64, 128x128)
- Output dimension: 128×16×16 = 32,768 (for 128x128 input)

### Decoders

#### RNN Decoder
- LSTM-based sequence decoder
- Configurable number of layers
- Teacher forcing during training

#### Transformer Decoder
- Multi-head attention mechanism
- Positional encoding
- Causal masking for autoregressive generation

### Symbol Mapping
- 45-token vocabulary
- Special tokens: `<PAD>`, `<START>`, `<END>`, `<UNK>`
- Bidirectional mapping between IDs and LaTeX symbols

## 🔄 Training Pipeline

### Features
- **Logging**: Comprehensive logging with timestamps
- **Checkpointing**: Automatic model saving
- **Validation**: Regular validation with accuracy metrics
- **Best Model**: Automatic saving of best performing model
- **Flexible Data**: Support for real and dummy datasets

### Training Process
1. Load configuration from `configs/config.yaml`
2. Initialize model and decoder (if enabled)
3. Create data loaders (Kaggle or dummy)
4. Train for specified epochs
5. Validate and save checkpoints
6. Log progress and metrics

## 🚀 Development

### Adding New Symbols

Edit `src/symbol_map.py`:
```python
self.symbol_map = {
    # ... existing symbols ...
    45: '\\new_symbol',  # Add new symbol
}
```

### Adding New Decoder Types

1. Create a new decoder class inheriting from `BaseDecoder`
2. Implement the `forward` method
3. Add the decoder type to the factory function in `src/decoder.py`

### Customizing Training

Modify `train.py` to add:
- Custom loss functions
- Additional metrics
- Learning rate scheduling
- Data augmentation

## 📝 Logging

Training logs are saved to `logs/` directory with timestamps:
- Training progress
- Loss and accuracy metrics
- Model checkpoints
- Error messages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in config.yaml
2. **Dataset not found**: Check data paths in config.yaml
3. **Import errors**: Ensure all dependencies are installed
4. **Model loading errors**: Check model file paths and formats

### Getting Help

- Check the logs in `logs/` directory
- Run tests to verify installation: `pytest tests/ -v`
- Use dummy data for testing: Set appropriate config values

---

**Ready to convert handwritten math to LaTeX! 🧮➡️📝**
