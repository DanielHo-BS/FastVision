import torch
import time
import numpy as np
import onnx
import onnxruntime as ort
import argparse
import json
from init_model import setup_env, init_model

class Model:
    def __init__(self, args):
        # Initialize model and environment
        self.device = setup_env(args.device)
        self.model_name = args.model
        self.model = init_model(self.model_name, self.device)
        self.onnx_model = None
        self.ort_session = None
        self.idx2label = None
        self.onnx_filename = f"{self.model_name}.onnx"
        
        # Load ImageNet class index
        self._load_imagenet_class_index()
        
        # Set model to evaluation mode
        try:
            print(f"Using device: {self.device}")
            self.model.eval()
        except Exception as e:
            print(f"Error setting model to eval mode: {e}")

    def _load_imagenet_class_index(self):
        """Load ImageNet class index."""
        with open("imagenet_class_index.json", "r") as f:
            self.class_idx = json.load(f)
        self.idx2label = [self.class_idx[str(k)][1] for k in range(len(self.class_idx))]

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

    def predict(self, input_tensor):
        """
        Run PyTorch inference and measure performance.
        
        Args:
            input_tensor: Input tensor for the model
        """
        with torch.no_grad():
            output = self.model(input_tensor)
            return self.idx2label[output.argmax(dim=1)]

    def convert_to_onnx(self, input_tensor = None):
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            input_tensor: Sample input tensor for tracing
        """
        
        if input_tensor is None:
            input_tensor = torch.randn(1, 3, 224, 224).to(self.device)
        
        try:
            # Export PyTorch model to ONNX format with a unique filename based on the model name
            torch.onnx.export(
                self.model,           # PyTorch model
                input_tensor,         # Sample input
                self.onnx_filename,   # Output file, unique to this model
                verbose=False         # Don't print detailed export info
            )
            
            # Load and validate the ONNX model
            self.onnx_model = onnx.load(self.onnx_filename)
            onnx.checker.check_model(self.onnx_model)
            print(f"ONNX model exported and validated successfully: {self.onnx_filename}")
            
            # Initialize ONNX runtime session
            self.ort_session = self._init_ort_session()
            
        except Exception as e:
            print(f"Error converting to ONNX: {e}")

    def _init_ort_session(self):
        """Initialize ONNX Runtime session for inference."""
        providers = ["CUDAExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
        return ort.InferenceSession(self.onnx_filename, providers=providers)

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

    def predict_onnx(self, input_tensor):
        """
        Run ONNX inference and measure performance.
        
        Args:
            input_tensor: Input tensor for the model
        """
        input_numpy = input_tensor.cpu().numpy()
        onnx_input = {self.ort_session.get_inputs()[0].name: input_numpy}
        with torch.no_grad():
            output = self.ort_session.run(None, onnx_input)
            # Get the top 1 prediction from numpy array
            top_1_pred = np.argmax(output[0])
            return self.idx2label[top_1_pred]

def main(args):
    # Create model instance
    model = Model(args)
    
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


    # Run PyTorch prediction
    pytorch_output = model.predict(dummy_input)
    print(f"PyTorch output: {pytorch_output}")
    
    # Run ONNX prediction
    onnx_output = model.predict_onnx(dummy_input)
    print(f"ONNX output: {onnx_output}")
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch to ONNX conversion and inference time comparison')
    parser.add_argument('--model', type=str, default='resnet18', required=False,
                        help='Model to use (resnet18, resnet50, vgg16, vision_transformer, etc.)')
    parser.add_argument('--device', type=str, default='cpu', required=False,
                        help='Device to run the model on (cpu, gpu, etc.)')
    args = parser.parse_args()
    main(args)