// QuickStart - Basic FusionML example
// Shows tensor creation, neural network, and training loop

import Foundation
import FusionML

@main
struct QuickStart {
    static func main() throws {
        print("🔥 FusionML Quick Start")
        print("=" .padding(toLength: 50, withPad: "=", startingAt: 0))
        
        Fusion.initialize()
        
        // 1. Create tensors
        print("\n📐 Tensor Creation:")
        let x = try Fusion.rand([2, 3])
        print("   Fusion.rand([2, 3]): \(x.toArray().prefix(3))")
        
        // 2. Build a simple model
        print("\n🧠 Neural Network:")
        let model = Fusion.nn.sequential(
            try Fusion.nn.linear(10, 5),
            Fusion.nn.relu()
        )
        print("   Sequential(Linear(10, 5), ReLU)")
        
        // 3. Forward pass
        let input = GradTensor(try Fusion.rand([4, 10]), requiresGrad: false)
        let output = try model.forward(input)
        print("   Input: [4, 10] → Output: \(output.shape)")
        
        // 4. Optimizer
        let optimizer = Fusion.optim.adam(model.parameters(), lr: 0.01)
        print("\n⚡ Optimizer: Adam(lr=0.01)")
        
        // 5. Simple training step
        print("\n🏋️ Training Step:")
        optimizer.zeroGrad()
        let target = try Fusion.random.randint(0, 5, [4])
        let loss = try Fusion.nn.functional.crossEntropy(output, target)
        print("   Loss: \(loss.data.toArray()[0])")
        
        try Fusion.autograd.backward(loss)
        try optimizer.step()
        print("   Backward ✓ | Step ✓")
        
        print("\n✅ FusionML Quick Start Complete!")
    }
}
