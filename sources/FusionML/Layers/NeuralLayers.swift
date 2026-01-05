// NeuralLayers - High-level Neural Network Layers
// Each layer is intelligently routed to optimal hardware

import Foundation
import Accelerate

/// Protocol for neural network layers
public protocol Layer {
    func forward(_ input: Tensor) throws -> Tensor
}

/// Linear layer: y = x @ W + b
public final class Linear: Layer {
    public let weight: Tensor  // [in_features, out_features]
    public let bias: Tensor?   // [out_features]
    
    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true) throws {
        // Xavier-like initialization (using random and will scale later)
        self.weight = try Tensor.random([inFeatures, outFeatures])
        self.bias = bias ? try Tensor.zeros([1, outFeatures]) : nil
    }
    
    public func forward(_ input: Tensor) throws -> Tensor {
        // Route matmul to best backend
        var output = try IntelligentRouter.shared.matmul(input, weight)
        if let bias = bias {
            output = try IntelligentRouter.shared.add(output, bias)
        }
        return output
    }
}

/// ReLU activation
public final class ReLU: Layer {
    public init() {}
    
    public func forward(_ input: Tensor) throws -> Tensor {
        try IntelligentRouter.shared.relu(input)
    }
}

/// GELU activation
public final class GELU: Layer {
    public init() {}
    
    public func forward(_ input: Tensor) throws -> Tensor {
        try IntelligentRouter.shared.gelu(input)
    }
}

/// Sequential container
public final class Sequential: Layer {
    public let layers: [Layer]
    
    public init(_ layers: Layer...) {
        self.layers = layers
    }
    
    public init(_ layers: [Layer]) {
        self.layers = layers
    }
    
    public func forward(_ input: Tensor) throws -> Tensor {
        var x = input
        for layer in layers {
            x = try layer.forward(x)
        }
        return x
    }
}

/// Simple MLP (Multi-Layer Perceptron)
public final class MLP: Layer {
    private let layers: Sequential
    
    public init(inputSize: Int, hiddenSize: Int, outputSize: Int) throws {
        layers = Sequential(
            try Linear(inFeatures: inputSize, outFeatures: hiddenSize),
            GELU(),
            try Linear(inFeatures: hiddenSize, outFeatures: outputSize)
        )
    }
    
    public func forward(_ input: Tensor) throws -> Tensor {
        try layers.forward(input)
    }
}

/// Transformer-style attention (simplified)
public final class SelfAttention: Layer {
    private let queryProj: Linear
    private let keyProj: Linear
    private let valueProj: Linear
    private let outProj: Linear
    private let headDim: Int
    private let scale: Float
    
    public init(embedDim: Int, numHeads: Int = 1) throws {
        self.headDim = embedDim / numHeads
        self.scale = 1.0 / sqrt(Float(headDim))
        
        queryProj = try Linear(inFeatures: embedDim, outFeatures: embedDim, bias: false)
        keyProj = try Linear(inFeatures: embedDim, outFeatures: embedDim, bias: false)
        valueProj = try Linear(inFeatures: embedDim, outFeatures: embedDim, bias: false)
        outProj = try Linear(inFeatures: embedDim, outFeatures: embedDim)
    }
    
    public func forward(_ input: Tensor) throws -> Tensor {
        let q = try queryProj.forward(input)
        let k = try keyProj.forward(input)
        let v = try valueProj.forward(input)
        
        // Attention: softmax(Q @ K^T / sqrt(d)) @ V
        // For simplicity, using matmul (real impl would transpose K)
        var scores = try IntelligentRouter.shared.matmul(q, k)
        
        // Scale
        let scoresData = scores.toArray()
        let scaledPtr = scores.buffer.pointer.bindMemory(to: Float.self, capacity: scores.count)
        for i in 0..<scores.count {
            scaledPtr[i] = scoresData[i] * scale
        }
        
        // Softmax (simplified - per row)
        // ... (would need proper implementation)
        
        let attended = try IntelligentRouter.shared.matmul(scores, v)
        return try outProj.forward(attended)
    }
}
