"""
Base class for neural network-based ML decoders.

Provides PyTorch and ONNX model loading infrastructure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from ..base import BaseMLDecoder


class NeuralNetworkDecoder(BaseMLDecoder):
    """
    Base class for neural network-based enhancement decoders.
    
    Handles model loading (PyTorch/ONNX), preprocessing, and postprocessing.
    Subclasses should implement model-specific inference logic.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        **config: Any
    ) -> None:
        """
        Initialize neural network decoder.
        
        Args:
            model_path: Path to model file (.pth for PyTorch, .onnx for ONNX)
            device: Device to run inference on ('cpu' or 'cuda')
            **config: Additional configuration
        """
        super().__init__(**config)
        self.model_path = model_path
        self.device = device
        self.model = None
        self.model_type: Optional[str] = None  # 'pytorch' or 'onnx'
        self.onnx_session = None
        
    def setup(self) -> None:
        """Load model from file."""
        super().setup()
        if self.model_path:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load model from file (PyTorch or ONNX)."""
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        model_path = Path(self.model_path)
        suffix = model_path.suffix.lower()
        
        if suffix == '.pth' or suffix == '.pt':
            self._load_pytorch_model(model_path)
        elif suffix == '.onnx':
            self._load_onnx_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {suffix}. Use .pth/.pt (PyTorch) or .onnx (ONNX)")
    
    def _load_pytorch_model(self, model_path: Path) -> None:
        """Load PyTorch model from .pth/.pt file."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        
        self.model_type = 'pytorch'
        
        # Load model
        # Note: For now, we expect the model to be loaded as a state dict or full model
        # Subclasses should override this if they need custom loading logic
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Check if it's a state dict or full model
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                self.model = checkpoint['model']
            elif 'state_dict' in checkpoint:
                # If only state_dict, subclasses need to provide model architecture
                self.model = checkpoint
            else:
                # Assume it's a state dict directly
                self.model = checkpoint
        else:
            # Assume it's a model object
            self.model = checkpoint
        
        if hasattr(self.model, 'eval'):
            self.model.eval()
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
    
    def _load_onnx_model(self, model_path: Path) -> None:
        """Load ONNX model from .onnx file."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime not installed. Install with: pip install onnxruntime")
        
        self.model_type = 'onnx'
        
        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda':
            try:
                providers.insert(0, 'CUDAExecutionProvider')
            except Exception:
                pass  # Fall back to CPU if CUDA not available
        
        self.onnx_session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # Store input/output names for convenience
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name
        self.onnx_output_name = self.onnx_session.get_outputs()[0].name
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for neural network input.
        
        Default implementation:
        - Converts BGR to RGB
        - Normalizes to [0, 1] range
        - Converts to float32
        - Adds batch dimension if needed
        
        Subclasses can override for custom preprocessing.
        
        Args:
            frame: Input frame in BGR format (H, W, 3), uint8
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # BGR to RGB
        rgb = frame[:, :, ::-1].copy()
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension: (H, W, 3) -> (1, H, W, 3)
        if len(normalized.shape) == 3:
            normalized = np.expand_dims(normalized, axis=0)
        
        # Convert to CHW format if needed (depends on model)
        # Most PyTorch models expect (B, C, H, W)
        if len(normalized.shape) == 4 and normalized.shape[-1] == 3:
            normalized = np.transpose(normalized, (0, 3, 1, 2))
        
        return normalized
    
    def postprocess(self, output: np.ndarray) -> np.ndarray:
        """
        Postprocess neural network output to BGR frame.
        
        Default implementation:
        - Removes batch dimension if present
        - Converts from CHW to HWC if needed
        - Denormalizes from [0, 1] to [0, 255]
        - Converts RGB to BGR
        - Clips to valid range [0, 255]
        - Converts to uint8
        
        Subclasses can override for custom postprocessing.
        
        Args:
            output: Model output tensor
            
        Returns:
            Enhanced frame in BGR format (H, W, 3), uint8
        """
        # Remove batch dimension if present: (1, C, H, W) -> (C, H, W)
        if len(output.shape) == 4:
            output = output[0]
        
        # Convert from CHW to HWC if needed: (C, H, W) -> (H, W, C)
        if len(output.shape) == 3 and output.shape[0] == 3:
            output = np.transpose(output, (1, 2, 0))
        
        # Denormalize: [0, 1] -> [0, 255]
        denormalized = output * 255.0
        
        # Clip to valid range
        clipped = np.clip(denormalized, 0, 255)
        
        # Convert to uint8
        uint8_frame = clipped.astype(np.uint8)
        
        # RGB to BGR
        bgr_frame = uint8_frame[:, :, ::-1].copy()
        
        return bgr_frame
    
    def _infer_pytorch(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run PyTorch inference."""
        import torch
        
        # Convert numpy to torch tensor
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Convert back to numpy
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        
        return output
    
    def _infer_onnx(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        if self.onnx_session is None:
            raise RuntimeError("ONNX model not loaded")
        
        # Run inference
        outputs = self.onnx_session.run(
            [self.onnx_output_name],
            {self.onnx_input_name: input_tensor}
        )
        
        return outputs[0]
    
    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run model inference.
        
        Args:
            input_tensor: Preprocessed input tensor
            
        Returns:
            Model output tensor
        """
        if self.model_type == 'pytorch':
            return self._infer_pytorch(input_tensor)
        elif self.model_type == 'onnx':
            return self._infer_onnx(input_tensor)
        else:
            raise RuntimeError("Model not loaded. Call setup() first.")
    
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame using neural network.
        
        Default implementation: preprocess -> infer -> postprocess
        Subclasses can override for custom behavior.
        
        Args:
            frame: Input frame in BGR format (H, W, 3), uint8
            
        Returns:
            Enhanced frame in BGR format (H, W, 3), uint8
        """
        self.ensure_setup()
        
        # Preprocess
        input_tensor = self.preprocess(frame)
        
        # Inference
        output_tensor = self.infer(input_tensor)
        
        # Postprocess
        enhanced_frame = self.postprocess(output_tensor)
        
        return enhanced_frame
    
    def teardown(self) -> None:
        """Release model resources."""
        super().teardown()
        self.model = None
        self.onnx_session = None



