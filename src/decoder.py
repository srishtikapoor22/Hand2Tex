import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class BaseDecoder(nn.Module):
    """Base decoder class that defines the interface for all decoders."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, max_length: int = 50):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_length = max_length
        
    def forward(self, encoder_output: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            encoder_output: Output from the encoder [batch_size, input_dim]
            target: Target sequence for training [batch_size, seq_len] (optional)
            
        Returns:
            Decoder output [batch_size, seq_len, num_classes]
        """
        raise NotImplementedError("Subclasses must implement forward method")


class RNNDecoder(BaseDecoder):
    """RNN-based decoder using LSTM."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, max_length: int = 50, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim, num_classes, max_length)
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Project encoder output to decoder hidden dimension
        self.encoder_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, num_classes)
        
        # Start token embedding
        self.start_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(self, encoder_output: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = encoder_output.size(0)
        
        # Project encoder output
        encoder_projected = self.encoder_projection(encoder_output).unsqueeze(1)
        
        # Initialize decoder state
        h0 = encoder_projected.transpose(0, 1).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        if self.training and target is not None:
            # Teacher forcing during training
            return self._forward_training(encoder_projected, target, h0, c0)
        else:
            # Autoregressive decoding during inference
            return self._forward_inference(encoder_projected, h0, c0)
    
    def _forward_training(self, encoder_projected: torch.Tensor, target: torch.Tensor, 
                         h0: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
        seq_len = target.size(1)
        
        # Create input sequence with start token
        start_tokens = self.start_embedding.repeat(encoder_projected.size(0), 1, 1)
        decoder_input = torch.cat([start_tokens, encoder_projected], dim=1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(decoder_input, (h0, c0))
        
        # Project to output classes
        outputs = self.output_projection(lstm_out)
        
        return outputs[:, 1:seq_len+1]  # Remove start token from output
    
    def _forward_inference(self, encoder_projected: torch.Tensor, h0: torch.Tensor, 
                          c0: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_projected.size(0)
        outputs = []
        
        # Start with encoder output
        current_input = encoder_projected
        h, c = h0, c0
        
        for step in range(self.max_length):
            # LSTM step
            lstm_out, (h, c) = self.lstm(current_input, (h, c))
            
            # Project to output classes
            step_output = self.output_projection(lstm_out)
            outputs.append(step_output)
            
            # Use predicted token as next input (greedy decoding)
            predicted_tokens = torch.argmax(step_output, dim=-1)
            # In a real implementation, you'd embed these tokens
            # For now, we'll use a simple projection
            current_input = self.encoder_projection(
                torch.zeros(batch_size, 1, self.input_dim, device=encoder_projected.device)
            )
        
        return torch.cat(outputs, dim=1)


class TransformerDecoder(BaseDecoder):
    """Transformer-based decoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, max_length: int = 50,
                 num_layers: int = 6, num_heads: int = 8, dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim, num_classes, max_length)
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Project encoder output to decoder dimension
        self.encoder_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_length, hidden_dim))
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, num_classes)
        
        # Start token embedding
        self.start_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(self, encoder_output: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = encoder_output.size(0)
        
        # Project encoder output
        encoder_projected = self.encoder_projection(encoder_output).unsqueeze(1)
        
        if self.training and target is not None:
            # Teacher forcing during training
            return self._forward_training(encoder_projected, target)
        else:
            # Autoregressive decoding during inference
            return self._forward_inference(encoder_projected)
    
    def _forward_training(self, encoder_projected: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        seq_len = target.size(1)
        
        # Create decoder input with start token
        start_tokens = self.start_embedding.repeat(encoder_projected.size(0), 1, 1)
        decoder_input = torch.cat([start_tokens, encoder_projected], dim=1)
        
        # Add positional encoding
        decoder_input = decoder_input + self.pos_encoding[:, :decoder_input.size(1)]
        
        # Create causal mask for autoregressive decoding
        mask = self._generate_causal_mask(decoder_input.size(1))
        
        # Transformer decoder forward pass
        transformer_out = self.transformer_decoder(
            tgt=decoder_input,
            memory=encoder_projected,
            tgt_mask=mask
        )
        
        # Project to output classes
        outputs = self.output_projection(transformer_out)
        
        return outputs[:, 1:seq_len+1]  # Remove start token from output
    
    def _forward_inference(self, encoder_projected: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_projected.size(0)
        outputs = []
        
        # Start with encoder output
        current_input = encoder_projected
        decoder_input = self.start_embedding.repeat(batch_size, 1, 1)
        
        for step in range(self.max_length):
            # Add positional encoding
            pos_enc = self.pos_encoding[:, :decoder_input.size(1)]
            decoder_input_with_pos = decoder_input + pos_enc
            
            # Create causal mask
            mask = self._generate_causal_mask(decoder_input_with_pos.size(1))
            
            # Transformer decoder step
            transformer_out = self.transformer_decoder(
                tgt=decoder_input_with_pos,
                memory=encoder_projected,
                tgt_mask=mask
            )
            
            # Get last token output
            last_output = transformer_out[:, -1:]
            step_output = self.output_projection(last_output)
            outputs.append(step_output)
            
            # Add predicted token to decoder input
            predicted_tokens = torch.argmax(step_output, dim=-1)
            # In a real implementation, you'd embed these tokens
            # For now, we'll use a simple embedding
            token_embedding = torch.zeros(batch_size, 1, self.hidden_dim, device=encoder_projected.device)
            decoder_input = torch.cat([decoder_input, token_embedding], dim=1)
        
        return torch.cat(outputs, dim=1)
    
    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask


def create_decoder(decoder_type: str, input_dim: int, hidden_dim: int, num_classes: int, 
                  max_length: int = 50, **kwargs) -> BaseDecoder:
    """
    Factory function to create decoder instances.
    
    Args:
        decoder_type: Type of decoder ('rnn' or 'transformer')
        input_dim: Input dimension from encoder
        hidden_dim: Hidden dimension for decoder
        num_classes: Number of output classes
        max_length: Maximum sequence length
        **kwargs: Additional arguments for specific decoder types
        
    Returns:
        Decoder instance
    """
    if decoder_type.lower() == 'rnn':
        return RNNDecoder(input_dim, hidden_dim, num_classes, max_length, **kwargs)
    elif decoder_type.lower() == 'transformer':
        return TransformerDecoder(input_dim, hidden_dim, num_classes, max_length, **kwargs)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}. Use 'rnn' or 'transformer'")
