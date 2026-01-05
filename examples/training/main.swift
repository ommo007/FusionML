// TrainingExample - Full training loop with FusionML
// Shows dataset, model, optimizer, and training

import Foundation
import FusionML

@main
struct TrainingExample {
    static func main() throws {
        print("🔥 FusionML Training Example")
        print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
        
        Fusion.initialize()
        
        // Configuration
        let inputSize = 64
        let hiddenSize = 128
        let outputSize = 10
        let batchSize = 32
        let numSamples = 320
        let epochs = 10
        
        print("\n📊 Config: \(inputSize)→\(hiddenSize)→\(outputSize), batch=\(batchSize)")
        
        // Create dataset
        let X = try Fusion.rand([numSamples, inputSize])
        let Y = try Fusion.random.randint(0, outputSize, [numSamples, 1])
        let dataset = Fusion.data.tensorDataset(data: X, labels: Y)
        let loader = Fusion.data.dataLoader(dataset, batchSize: batchSize, shuffle: true)
        
        // Build model
        let model = Fusion.nn.sequential(
            try Fusion.nn.linear(inputSize, hiddenSize),
            Fusion.nn.relu(),
            try Fusion.nn.linear(hiddenSize, hiddenSize),
            Fusion.nn.relu(),
            try Fusion.nn.linear(hiddenSize, outputSize)
        )
        
        let optimizer = Fusion.optim.adam(model.parameters(), lr: 0.01)
        
        print("\n🏋️ Training...")
        print("-" .padding(toLength: 60, withPad: "-", startingAt: 0))
        
        var losses: [Float] = []
        let start = CFAbsoluteTimeGetCurrent()
        
        for epoch in 0..<epochs {
            var epochLoss: Float = 0
            var count = 0
            
            for batch in loader {
                let (batchX, batchY) = try collateTensors(batch)
                
                optimizer.zeroGrad()
                
                let input = GradTensor(batchX, requiresGrad: false)
                let output = try model.forward(input)
                
                let flatY = try Tensor(shape: [batchY.shape[0]], dtype: .float32)
                memcpy(flatY.buffer.pointer, batchY.buffer.pointer, batchY.count * 4)
                
                let loss = try Fusion.nn.functional.crossEntropy(output, flatY)
                try Fusion.autograd.backward(loss)
                try optimizer.step()
                
                epochLoss += loss.data.toArray()[0]
                count += 1
            }
            
            let avgLoss = epochLoss / Float(count)
            losses.append(avgLoss)
            print("   Epoch \(epoch + 1)/\(epochs) | Loss: \(String(format: "%.4f", avgLoss))")
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - start
        
        print("-" .padding(toLength: 60, withPad: "-", startingAt: 0))
        print("\n📈 Results:")
        print("   Initial: \(String(format: "%.4f", losses.first ?? 0))")
        print("   Final:   \(String(format: "%.4f", losses.last ?? 0))")
        print("   Time:    \(String(format: "%.2f", duration))s")
        
        print("\n✅ Training Complete!")
    }
}
