import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import time
import numpy as np
import onnx
import onnxruntime as ort

class Model:
    def __init__(self):
        # Initialize model and environment
        self.device = self._setup_env()
        self.model = self._init_model()
        self.onnx_model = None
        self.ort_session = None
        
        # Set model to evaluation mode
        try:
            print(f"Using device: {self.device}")
            self.model.eval()
        except Exception as e:
            print(f"Error setting model to eval mode: {e}")

    def _setup_env(self):
        """Set up the computing environment. Always use CPU for consistency."""
        return torch.device("cpu")

    def _init_model(self):
        """Initialize ResNet18 model with ImageNet weights."""
        return models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(self.device)
    
    def inference(self, input_tensor):
        """
        Run PyTorch inference and measure performance.
        
        Args:
            input_tensor: Input tensor for the model
            
        Returns:
            float: Average inference time per run
        """
        with torch.no_grad():
            # Run inference 100 times and measure total time
            start_time = time.time()
            for _ in range(100):
                _ = self.model(input_tensor)
            end_time = time.time()

        # Calculate and print average inference time
        avg_time = (end_time - start_time) / 100
        print(f"PyTorch average inference time: {avg_time:.6f} seconds per run")
        return avg_time

    def convert_to_onnx(self, input_tensor):
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            input_tensor: Sample input tensor for tracing
        """
        try:
            # Export PyTorch model to ONNX format
            torch.onnx.export(
                self.model,           # PyTorch model
                input_tensor,         # Sample input
                "model.onnx",         # Output file
                verbose=False         # Don't print detailed export info
            )
            
            # Load and validate the ONNX model
            self.onnx_model = onnx.load("model.onnx")
            onnx.checker.check_model(self.onnx_model)
            print("ONNX model exported and validated successfully")
            
            # Initialize ONNX runtime session
            self.ort_session = self._init_ort_session()
            
        except Exception as e:
            print(f"Error converting to ONNX: {e}")

    def _init_ort_session(self):
        """Initialize ONNX Runtime session for inference."""
        return ort.InferenceSession("model.onnx")

    def inference_onnx(self, input_tensor):
        """
        Run ONNX inference and measure performance.
        
        Args:
            input_tensor: Input tensor for the model
            
        Returns:
            float: Average inference time per run
        """
        # Convert input tensor to numpy array for ONNX runtime
        input_numpy = input_tensor.cpu().numpy()
        # Prepare input dictionary for ONNX runtime
        onnx_input = {self.ort_session.get_inputs()[0].name: input_numpy}

        # Run inference 100 times and measure total time
        start_time = time.time()
        for _ in range(100):
            _ = self.ort_session.run(None, onnx_input)
        end_time = time.time()

        # Calculate and print average inference time
        avg_time = (end_time - start_time) / 100
        print(f"ONNX average inference time: {avg_time:.6f} seconds per run")
        return avg_time

def main():
    # Create model instance
    model = Model()
    
    # Create sample input tensor
    dummy_input = torch.randn(1, 3, 224, 224).to(model.device)
    
    # Run PyTorch inference and get timing
    pytorch_time = model.inference(dummy_input)
    
    # Convert model to ONNX format
    model.convert_to_onnx(dummy_input)
    
    # Run ONNX inference and get timing
    onnx_time = model.inference_onnx(dummy_input)
    
    # Compare performance
    speedup = pytorch_time / onnx_time
    print(f"\nPerformance Comparison:")
    print(f"ONNX is {speedup:.2f}x faster than PyTorch")
    
if __name__ == "__main__":
    main()