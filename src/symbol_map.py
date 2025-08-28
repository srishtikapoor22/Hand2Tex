from typing import Dict, List, Optional
import torch

class SymbolMapper:
    """Maps model predictions to LaTeX symbols."""
    
    def __init__(self):
        # Define the mapping of symbol IDs to LaTeX symbols
        self.symbol_map = {
            0: '+',      # Plus
            1: '-',      # Minus
            2: '=',      # Equals
            3: '\\times', # Multiplication
            4: '\\div',  # Division
            5: '\\frac', # Fraction
            6: '\\sqrt', # Square root
            7: '\\alpha', # Alpha
            8: '\\beta',  # Beta
            9: '\\pi',    # Pi
            10: '\\int',  # Integral
            11: '\\sum',  # Summation
            12: '\\infty', # Infinity
            13: '\\leq',  # Less than or equal
            14: '\\geq',  # Greater than or equal
            15: '\\cdot', # Dot product
            16: '\\sin',  # Sine
            17: '\\cos',  # Cosine
            18: '\\tan',  # Tangent
            19: '\\log',  # Logarithm
            20: '(',      # Left parenthesis
            21: ')',      # Right parenthesis
            22: '[',      # Left bracket
            23: ']',      # Right bracket
            24: '{',      # Left brace
            25: '}',      # Right brace
            26: 'x',      # Variable x
            27: 'y',      # Variable y
            28: 'z',      # Variable z
            29: 'n',      # Variable n
            30: '0',      # Digit 0
            31: '1',      # Digit 1
            32: '2',      # Digit 2
            33: '3',      # Digit 3
            34: '4',      # Digit 4
            35: '5',      # Digit 5
            36: '6',      # Digit 6
            37: '7',      # Digit 7
            38: '8',      # Digit 8
            39: '9',      # Digit 9
            40: ' ',      # Space
            41: '<PAD>',  # Padding token
            42: '<START>', # Start token
            43: '<END>',   # End token
            44: '<UNK>'    # Unknown token
        }
        
        # Reverse mapping for training
        self.reverse_map = {v: k for k, v in self.symbol_map.items()}
        
        # Special tokens
        self.pad_token_id = 41
        self.start_token_id = 42
        self.end_token_id = 43
        self.unk_token_id = 44
        
        # Vocabulary size
        self.vocab_size = len(self.symbol_map)
    
    def id_to_symbol(self, token_id: int) -> str:
        """
        Convert token ID to LaTeX symbol.
        
        Args:
            token_id: Token ID from model prediction
            
        Returns:
            LaTeX symbol string
        """
        return self.symbol_map.get(token_id, '<UNK>')
    
    def symbol_to_id(self, symbol: str) -> int:
        """
        Convert LaTeX symbol to token ID.
        
        Args:
            symbol: LaTeX symbol string
            
        Returns:
            Token ID
        """
        return self.reverse_map.get(symbol, self.unk_token_id)
    
    def ids_to_latex(self, token_ids: List[int]) -> str:
        """
        Convert a sequence of token IDs to LaTeX string.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            LaTeX string
        """
        # Remove special tokens and convert to symbols
        symbols = []
        for token_id in token_ids:
            if token_id in [self.pad_token_id, self.start_token_id, self.end_token_id]:
                continue
            symbols.append(self.id_to_symbol(token_id))
        
        # Join symbols into LaTeX string
        latex_str = ''.join(symbols)
        
        # Clean up the string
        latex_str = latex_str.strip()
        
        return latex_str
    
    def tensor_to_latex(self, predictions: torch.Tensor, method: str = 'greedy') -> str:
        """
        Convert model predictions tensor to LaTeX string.
        
        Args:
            predictions: Model output tensor [batch_size, seq_len, vocab_size] or [seq_len, vocab_size]
            method: Decoding method ('greedy' or 'beam')
            
        Returns:
            LaTeX string
        """
        if predictions.dim() == 3:
            # Remove batch dimension if present
            predictions = predictions.squeeze(0)
        
        if method == 'greedy':
            # Greedy decoding: take argmax at each position
            token_ids = torch.argmax(predictions, dim=-1).tolist()
        else:
            raise ValueError(f"Unknown decoding method: {method}")
        
        return self.ids_to_latex(token_ids)
    
    def latex_to_ids(self, latex_str: str, max_length: int = 50) -> List[int]:
        """
        Convert LaTeX string to token IDs.
        
        Args:
            latex_str: LaTeX string
            max_length: Maximum sequence length
            
        Returns:
            List of token IDs
        """
        # Add start token
        token_ids = [self.start_token_id]
        
        # Convert each character to token ID
        for char in latex_str:
            token_id = self.symbol_to_id(char)
            token_ids.append(token_id)
        
        # Add end token
        token_ids.append(self.end_token_id)
        
        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(self.pad_token_id)
        
        return token_ids[:max_length]
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            'pad': self.pad_token_id,
            'start': self.start_token_id,
            'end': self.end_token_id,
            'unk': self.unk_token_id
        }


# Global symbol mapper instance
symbol_mapper = SymbolMapper()


def get_symbol_mapper() -> SymbolMapper:
    """Get the global symbol mapper instance."""
    return symbol_mapper


def create_latex_from_predictions(predictions: torch.Tensor, method: str = 'greedy') -> str:
    """
    Convenience function to convert model predictions to LaTeX.
    
    Args:
        predictions: Model output tensor
        method: Decoding method
        
    Returns:
        LaTeX string
    """
    return symbol_mapper.tensor_to_latex(predictions, method)


def create_predictions_from_latex(latex_str: str, max_length: int = 50) -> List[int]:
    """
    Convenience function to convert LaTeX string to token IDs.
    
    Args:
        latex_str: LaTeX string
        max_length: Maximum sequence length
        
    Returns:
        List of token IDs
    """
    return symbol_mapper.latex_to_ids(latex_str, max_length)
