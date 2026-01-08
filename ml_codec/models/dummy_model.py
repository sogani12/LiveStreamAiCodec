"""
Dummy PyTorch model for testing neural network infrastructure.

This is a simple identity model that passes frames through unchanged,
useful for verifying the model loading and inference pipeline works.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .neural_base import NeuralNetworkDecoder


class DummyNeuralDecoder(NeuralNetworkDecoder):
    """
    Dummy decoder that uses a simple identity PyTorch model.
    
    Useful for testing the neural network infrastructure without
    requiring a real trained model.
    """
    
    def __init__(self, model_path: Optional[str] = None, **config: Any) -> None:
        """
        Initialize dummy decoder.
        
        Args:
            model_path: Optional path to model file (if None, creates dummy model)
            **config: Additional configuration
        """
        super().__init__(model_path=model_path, **config)
        self._dummy_model = None
    
    def setup(self) -> None:
        """Create or load dummy model."""
        super().setup()
        
        if self.model is None and self.onnx_session is None:
            # Create a simple identity model if no model file provided
            self._create_dummy_model()
    
    def _create_dummy_model(self) -> None:
        """Create a simple identity PyTorch model for testing."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        
        class IdentityModel(nn.Module):
            """Simple identity model that passes input through unchanged."""
            def __init__(self):
                super().__init__()
                # Single 1x1 conv that acts as identity
                self.conv = nn.Conv2d(3, 3, kernel_size=1, bias=False)
                # Initialize to identity
                with torch.no_grad():
                    self.conv.weight.fill_(0)
                    for i in range(3):
                        self.conv.weight[i, i] = 1.0
            
            def forward(self, x):
                return self.conv(x)
        
        self.model = IdentityModel()
        self.model.eval()
        self.model_type = 'pytorch'
        
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
    
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame (identity operation for dummy model).
        
        This verifies the full pipeline: preprocess -> infer -> postprocess
        """
        self.ensure_setup()
        
        # Preprocess
        input_tensor = self.preprocess(frame)
        
        # Inference
        output_tensor = self.infer(input_tensor)
        
        # Postprocess
        enhanced_frame = self.postprocess(output_tensor)
        
        return enhanced_frame

