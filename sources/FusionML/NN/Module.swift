// Module - Base protocol for neural network modules
// Like PyTorch's nn.Module

import Foundation

// MARK: - Module Protocol

/// Base protocol for all neural network modules
public protocol Module: AnyObject {
    /// Forward pass
    func forward(_ input: GradTensor) throws -> GradTensor
    
    /// Get all trainable parameters
    func parameters() -> [GradTensor]
    
    /// Get named parameters
    func namedParameters() -> [(String, GradTensor)]
    
    /// Set training mode
    func train()
    
    /// Set evaluation mode
    func eval()
    
    /// Whether in training mode
    var training: Bool { get set }
}

// MARK: - Default Implementations

extension Module {
    public func train() {
        training = true
    }
    
    public func eval() {
        training = false
    }
}

// MARK: - GradLinear Layer

/// Linear layer with gradient tracking
public final class GradLinear: Module {
    public var training: Bool = true
    
    public let weight: GradTensor
    public let bias: GradTensor?
    public let inFeatures: Int
    public let outFeatures: Int
    
    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true) throws {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        
        // Xavier initialization
        let scale = sqrt(2.0 / Float(inFeatures + outFeatures))
        var w = try Tensor.random([inFeatures, outFeatures])
        let wPtr = w.buffer.pointer.bindMemory(to: Float.self, capacity: w.count)
        for i in 0..<w.count {
            wPtr[i] = (wPtr[i] - 0.5) * 2 * scale
        }
        self.weight = GradTensor(w, requiresGrad: true)
        
        if bias {
            self.bias = GradTensor(try Tensor.zeros([1, outFeatures]), requiresGrad: true)
        } else {
            self.bias = nil
        }
    }
    
    public func forward(_ input: GradTensor) throws -> GradTensor {
        var output = try GradTensor.matmul(input, weight)
        if let bias = bias {
            // Broadcast bias to match batch size
            let batchSize = input.shape[0]
            let biasData = bias.data.toArray()
            var broadcastedBias = try Tensor(shape: [batchSize, outFeatures], dtype: bias.data.dtype)
            let ptr = broadcastedBias.buffer.pointer.bindMemory(to: Float.self, capacity: broadcastedBias.count)
            
            for b in 0..<batchSize {
                for f in 0..<outFeatures {
                    ptr[b * outFeatures + f] = biasData[f]
                }
            }
            
            let biasGrad = GradTensor(broadcastedBias, requiresGrad: bias.requiresGrad)
            biasGrad.isLeaf = false
            output = try GradTensor.add(output, biasGrad)
        }
        return output
    }
    
    public func parameters() -> [GradTensor] {
        if let bias = bias {
            return [weight, bias]
        }
        return [weight]
    }
    
    public func namedParameters() -> [(String, GradTensor)] {
        var params: [(String, GradTensor)] = [("weight", weight)]
        if let bias = bias {
            params.append(("bias", bias))
        }
        return params
    }
}

// MARK: - Activation Layers

/// ReLU activation as a Module
public final class GradReLU: Module {
    public var training: Bool = true
    
    public init() {}
    
    public func forward(_ input: GradTensor) throws -> GradTensor {
        try input.relu()
    }
    
    public func parameters() -> [GradTensor] { [] }
    public func namedParameters() -> [(String, GradTensor)] { [] }
}

/// GELU activation as a Module
public final class GradGELU: Module {
    public var training: Bool = true
    
    public init() {}
    
    public func forward(_ input: GradTensor) throws -> GradTensor {
        try input.gelu()
    }
    
    public func parameters() -> [GradTensor] { [] }
    public func namedParameters() -> [(String, GradTensor)] { [] }
}

// MARK: - Sequential Container

/// Sequential container for chaining modules
public final class GradSequential: Module {
    public var training: Bool = true {
        didSet {
            for module in modules {
                module.training = training
            }
        }
    }
    
    public let modules: [Module]
    
    public init(_ modules: Module...) {
        self.modules = modules
    }
    
    public init(_ modules: [Module]) {
        self.modules = modules
    }
    
    public func forward(_ input: GradTensor) throws -> GradTensor {
        var x = input
        for module in modules {
            x = try module.forward(x)
        }
        return x
    }
    
    public func parameters() -> [GradTensor] {
        modules.flatMap { $0.parameters() }
    }
    
    public func namedParameters() -> [(String, GradTensor)] {
        var params: [(String, GradTensor)] = []
        for (i, module) in modules.enumerated() {
            for (name, param) in module.namedParameters() {
                params.append(("\(i).\(name)", param))
            }
        }
        return params
    }
}

// MARK: - Dropout

/// Dropout layer
public final class GradDropout: Module {
    public var training: Bool = true
    public let p: Float
    
    public init(p: Float = 0.5) {
        self.p = p
    }
    
    public func forward(_ input: GradTensor) throws -> GradTensor {
        guard training else { return input }
        
        // Create mask
        var mask = try Tensor(shape: input.shape, dtype: input.data.dtype)
        let maskPtr = mask.buffer.pointer.bindMemory(to: Float.self, capacity: mask.count)
        let scale = 1.0 / (1.0 - p)
        
        for i in 0..<mask.count {
            maskPtr[i] = Float.random(in: 0..<1) > p ? Float(scale) : 0
        }
        
        let maskTensor = GradTensor(mask, requiresGrad: false)
        return try GradTensor.mul(input, maskTensor)
    }
    
    public func parameters() -> [GradTensor] { [] }
    public func namedParameters() -> [(String, GradTensor)] { [] }
}

// MARK: - Layer Normalization

/// Layer normalization
public final class GradLayerNorm: Module {
    public var training: Bool = true
    
    public let normalizedShape: [Int]
    public let gamma: GradTensor
    public let beta: GradTensor
    public let eps: Float
    
    public init(normalizedShape: [Int], eps: Float = 1e-5) throws {
        self.normalizedShape = normalizedShape
        self.eps = eps
        
        let size = normalizedShape.reduce(1, *)
        self.gamma = GradTensor(try Tensor.ones([size]), requiresGrad: true)
        self.beta = GradTensor(try Tensor.zeros([size]), requiresGrad: true)
    }
    
    public func forward(_ input: GradTensor) throws -> GradTensor {
        // Simplified layer norm for last dimension
        let data = input.data.toArray()
        let lastDim = input.shape.last!
        let numGroups = input.count / lastDim
        
        var result = try Tensor(shape: input.shape, dtype: input.data.dtype)
        let resultPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
        let gammaData = gamma.data.toArray()
        let betaData = beta.data.toArray()
        
        for g in 0..<numGroups {
            let offset = g * lastDim
            
            // Compute mean
            var mean: Float = 0
            for i in 0..<lastDim {
                mean += data[offset + i]
            }
            mean /= Float(lastDim)
            
            // Compute variance
            var variance: Float = 0
            for i in 0..<lastDim {
                let diff = data[offset + i] - mean
                variance += diff * diff
            }
            variance /= Float(lastDim)
            
            // Normalize
            let std = sqrt(variance + eps)
            for i in 0..<lastDim {
                let normalized = (data[offset + i] - mean) / std
                resultPtr[offset + i] = normalized * gammaData[i % gammaData.count] + betaData[i % betaData.count]
            }
        }
        
        return GradTensor(result, requiresGrad: input.requiresGrad)
    }
    
    public func parameters() -> [GradTensor] { [gamma, beta] }
    public func namedParameters() -> [(String, GradTensor)] { [("gamma", gamma), ("beta", beta)] }
}
