#!/usr/bin/env python3
"""
FusionML Quick Start Example
"""

import sys
sys.path.insert(0, '.')

import fusionml as fml

def main():
    print("=" * 50)
    print("🔥 FusionML Python - Quick Start")
    print("=" * 50)
    
    fml.init()
    
    # 1. Tensor creation
    print("\n📐 Tensor Creation:")
    x = fml.rand(2, 3)
    print(f"   fml.rand(2, 3): {x.shape}")
    
    # 2. Build model
    print("\n🧠 Neural Network:")
    model = fml.nn.Sequential([
        fml.nn.Linear(10, 5),
        fml.nn.ReLU()
    ])
    print("   Sequential(Linear(10, 5), ReLU)")
    
    # 3. Forward pass
    input_tensor = fml.rand(4, 10)
    input_tensor.requires_grad = True
    output = model(input_tensor)
    print(f"   Input: {input_tensor.shape} → Output: {output.shape}")
    
    # 4. Optimizer
    print("\n⚡ Optimizer:")
    optimizer = fml.optim.Adam(model.parameters(), lr=0.01)
    print("   Adam(lr=0.01)")
    
    # 5. Training step
    print("\n🏋️ Training Step:")
    target = fml.Tensor([0, 1, 2, 3])
    loss = fml.nn.functional.cross_entropy(output, target)
    print(f"   Loss: {loss.item():.4f}")
    
    loss.backward()
    optimizer.step()
    print("   Backward ✓ | Step ✓")
    
    print("\n" + "=" * 50)
    print("✅ FusionML Python - Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
