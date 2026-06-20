import os
import torch
import coremltools as ct

class MatMulModel(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)

def main():
    os.makedirs("models", exist_ok=True)
    sizes = [512, 1024, 2048] # We can support 512, 1024 and 2048 to keep models light and compile fast
    
    for size in sizes:
        print(f"Generating CoreML matmul model for size {size}...")
        model = MatMulModel().eval()
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        traced = torch.jit.trace(model, (a, b))
        
        cml_model = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="a", shape=a.shape),
                ct.TensorType(name="b", shape=b.shape)
            ],
            compute_units=ct.ComputeUnit.ALL, # Allow ANE
            minimum_deployment_target=ct.target.macOS13
        )
        
        out_path = f"models/matmul_{size}.mlpackage"
        cml_model.save(out_path)
        print(f"Saved CoreML model to {out_path}")

if __name__ == "__main__":
    main()
