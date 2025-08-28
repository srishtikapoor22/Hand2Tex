import torch
import pytest
import tempfile
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import preprocess_image
from src.model import SimpleMathOCR
from src.decoder import RNNDecoder, TransformerDecoder, create_decoder
from src.symbol_map import SymbolMapper, get_symbol_mapper, create_latex_from_predictions


def test_basic_pipeline():
    """Test the basic pipeline with original model."""
    img = preprocess_image("data/sample.png")
    model = SimpleMathOCR(num_classes=5, use_decoder=False)
    out = model(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape[1] == 5


def test_model_with_decoder():
    """Test model with external decoder architecture."""
    img = preprocess_image("data/sample.png")
    model = SimpleMathOCR(use_decoder=True, input_size=128)
    encoder_output = model(img)
    assert isinstance(encoder_output, torch.Tensor)
    assert encoder_output.shape[1] == 128 * 16 * 16  # 32768


def test_model_encoder_output_dim():
    """Test that encoder output dimension is calculated correctly."""
    # Test 128x128 input
    model_128 = SimpleMathOCR(use_decoder=True, input_size=128)
    assert model_128.get_encoder_output_dim() == 128 * 16 * 16
    
    # Test 64x64 input
    model_64 = SimpleMathOCR(use_decoder=True, input_size=64)
    assert model_64.get_encoder_output_dim() == 128 * 8 * 8


def test_rnn_decoder():
    """Test RNN decoder functionality."""
    batch_size = 2
    input_dim = 100
    hidden_dim = 64
    num_classes = 45
    max_length = 20
    
    decoder = RNNDecoder(input_dim, hidden_dim, num_classes, max_length)
    
    # Test with encoder output
    encoder_output = torch.randn(batch_size, input_dim)
    output = decoder(encoder_output)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, max_length, num_classes)


def test_transformer_decoder():
    """Test Transformer decoder functionality."""
    batch_size = 2
    input_dim = 100
    hidden_dim = 64
    num_classes = 45
    max_length = 20
    
    decoder = TransformerDecoder(input_dim, hidden_dim, num_classes, max_length)
    
    # Test with encoder output
    encoder_output = torch.randn(batch_size, input_dim)
    output = decoder(encoder_output)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, max_length, num_classes)


def test_decoder_factory():
    """Test decoder factory function."""
    input_dim = 100
    hidden_dim = 64
    num_classes = 45
    
    # Test RNN decoder creation
    rnn_decoder = create_decoder('rnn', input_dim, hidden_dim, num_classes)
    assert isinstance(rnn_decoder, RNNDecoder)
    
    # Test Transformer decoder creation
    transformer_decoder = create_decoder('transformer', input_dim, hidden_dim, num_classes)
    assert isinstance(transformer_decoder, TransformerDecoder)
    
    # Test invalid decoder type
    with pytest.raises(ValueError):
        create_decoder('invalid', input_dim, hidden_dim, num_classes)


def test_symbol_mapper():
    """Test symbol mapping functionality."""
    mapper = SymbolMapper()
    
    # Test vocabulary size
    assert mapper.get_vocab_size() == 45
    
    # Test ID to symbol mapping
    assert mapper.id_to_symbol(0) == '+'
    assert mapper.id_to_symbol(1) == '-'
    assert mapper.id_to_symbol(2) == '='
    assert mapper.id_to_symbol(3) == '\\times'
    
    # Test symbol to ID mapping
    assert mapper.symbol_to_id('+') == 0
    assert mapper.symbol_to_id('-') == 1
    assert mapper.symbol_to_id('=') == 2
    assert mapper.symbol_to_id('\\times') == 3
    
    # Test unknown symbol
    assert mapper.symbol_to_id('unknown_symbol') == mapper.unk_token_id


def test_latex_conversion():
    """Test LaTeX string conversion."""
    mapper = SymbolMapper()
    
    # Test IDs to LaTeX
    token_ids = [42, 0, 1, 2, 43]  # START, +, -, =, END
    latex = mapper.ids_to_latex(token_ids)
    assert latex == '+-='
    
    # Test LaTeX to IDs
    latex_str = '+-='
    ids = mapper.latex_to_ids(latex_str, max_length=10)
    assert len(ids) == 10
    assert ids[0] == mapper.start_token_id
    assert ids[1] == 0  # '+'
    assert ids[2] == 1  # '-'
    assert ids[3] == 2  # '='
    assert ids[4] == mapper.end_token_id


def test_tensor_to_latex():
    """Test tensor to LaTeX conversion."""
    mapper = SymbolMapper()
    
    # Create dummy predictions tensor
    batch_size = 1
    seq_len = 5
    vocab_size = mapper.get_vocab_size()
    
    # Create predictions where each position predicts a specific token
    predictions = torch.zeros(batch_size, seq_len, vocab_size)
    predictions[0, 0, 0] = 1.0  # Predict '+'
    predictions[0, 1, 1] = 1.0  # Predict '-'
    predictions[0, 2, 2] = 1.0  # Predict '='
    predictions[0, 3, 26] = 1.0  # Predict 'x'
    predictions[0, 4, 30] = 1.0  # Predict '0'
    
    latex = mapper.tensor_to_latex(predictions)
    assert latex == '+-=x0'


def test_global_symbol_mapper():
    """Test global symbol mapper instance."""
    mapper = get_symbol_mapper()
    assert isinstance(mapper, SymbolMapper)
    
    # Test convenience function
    predictions = torch.zeros(1, 3, 45)
    predictions[0, 0, 0] = 1.0  # Predict '+'
    predictions[0, 1, 1] = 1.0  # Predict '-'
    predictions[0, 2, 2] = 1.0  # Predict '='
    
    latex = create_latex_from_predictions(predictions)
    assert latex == '+-='


def test_mapping_correctness():
    """Test that predicted IDs map to expected LaTeX symbols."""
    mapper = SymbolMapper()
    
    # Test a few key mathematical symbols
    test_cases = [
        (0, '+'),
        (1, '-'),
        (2, '='),
        (3, '\\times'),
        (4, '\\div'),
        (5, '\\frac'),
        (6, '\\sqrt'),
        (7, '\\alpha'),
        (8, '\\beta'),
        (9, '\\pi'),
        (10, '\\int'),
        (11, '\\sum'),
        (12, '\\infty'),
        (13, '\\leq'),
        (14, '\\geq'),
        (15, '\\cdot'),
        (16, '\\sin'),
        (17, '\\cos'),
        (18, '\\tan'),
        (19, '\\log')
    ]
    
    for token_id, expected_symbol in test_cases:
        symbol = mapper.id_to_symbol(token_id)
        assert symbol == expected_symbol, f"Token ID {token_id} should map to '{expected_symbol}', got '{symbol}'"


def test_inference_correctness():
    """Test inference correctness with dummy image and mocked pipeline."""
    # Create a dummy image tensor
    dummy_img = torch.randn(1, 1, 128, 128)
    
    # Test basic model
    basic_model = SimpleMathOCR(use_decoder=False, num_classes=45)
    with torch.no_grad():
        basic_output = basic_model(dummy_img)
    assert isinstance(basic_output, torch.Tensor)
    assert basic_output.shape[1] == 45
    
    # Test decoder model
    decoder_model = SimpleMathOCR(use_decoder=True, input_size=128)
    decoder = RNNDecoder(
        input_dim=decoder_model.get_encoder_output_dim(),
        hidden_dim=64,
        num_classes=45,
        max_length=10
    )
    
    with torch.no_grad():
        encoder_output = decoder_model(dummy_img)
        decoder_output = decoder(encoder_output)
        mapper = get_symbol_mapper()
        latex = mapper.tensor_to_latex(decoder_output)
    
    assert isinstance(latex, str)
    assert len(latex) >= 0  # Can be empty for untrained model


def test_special_tokens():
    """Test special token handling."""
    mapper = SymbolMapper()
    special_tokens = mapper.get_special_tokens()
    
    assert 'pad' in special_tokens
    assert 'start' in special_tokens
    assert 'end' in special_tokens
    assert 'unk' in special_tokens
    
    # Test that special tokens are handled correctly in conversion
    token_ids = [mapper.start_token_id, 0, 1, mapper.end_token_id, mapper.pad_token_id]
    latex = mapper.ids_to_latex(token_ids)
    assert latex == '+-'  # Special tokens should be removed


def test_model_architecture_compatibility():
    """Test that both model architectures work correctly."""
    # Test basic model
    basic_model = SimpleMathOCR(num_classes=45, use_decoder=False, input_size=128)
    dummy_input = torch.randn(1, 1, 128, 128)
    basic_output = basic_model(dummy_input)
    assert basic_output.shape == (1, 45)
    
    # Test decoder model
    decoder_model = SimpleMathOCR(num_classes=45, use_decoder=True, input_size=128)
    decoder_output = decoder_model(dummy_input)
    assert decoder_output.shape == (1, 128 * 16 * 16)  # Encoder output dimension


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__])
