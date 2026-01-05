// Fusion.nn - Neural Network Module
// Layers, containers, and functional API

import Foundation

extension Fusion {
    
    /// Neural network module - layers and functional operations
    public enum nn {
        
        // MARK: - Layers (type aliases to existing classes)
        
        /// Linear layer: y = x @ W + b
        public typealias Linear = GradLinear
        
        /// ReLU activation
        public typealias ReLU = GradReLU
        
        /// GELU activation
        public typealias GELU = GradGELU
        
        /// Dropout layer
        public typealias Dropout = GradDropout
        
        /// Layer normalization
        public typealias LayerNorm = GradLayerNorm
        
        /// Sequential container
        public typealias Sequential = GradSequential
        
        /// Module protocol - use directly Module
        
        // MARK: - Factory Methods
        
        /// Create Linear layer
        public static func linear(_ inFeatures: Int, _ outFeatures: Int, bias: Bool = true) throws -> GradLinear {
            try GradLinear(inFeatures: inFeatures, outFeatures: outFeatures, bias: bias)
        }
        
        /// Create ReLU activation
        public static func relu() -> GradReLU {
            GradReLU()
        }
        
        /// Create GELU activation  
        public static func gelu() -> GradGELU {
            GradGELU()
        }
        
        /// Create Dropout
        public static func dropout(_ p: Float = 0.5) -> GradDropout {
            GradDropout(p: p)
        }
        
        /// Create LayerNorm
        public static func layerNorm(_ normalizedShape: [Int], eps: Float = 1e-5) throws -> GradLayerNorm {
            try GradLayerNorm(normalizedShape: normalizedShape, eps: eps)
        }
        
        /// Create Sequential
        public static func sequential(_ modules: Module...) -> GradSequential {
            GradSequential(modules)
        }
    }
}

// MARK: - Fusion.nn.functional

extension Fusion.nn {
    
    /// Functional operations (stateless)
    public enum functional {
        
        // MARK: - Activations
        
        /// Apply ReLU activation
        public static func relu(_ input: GradTensor) throws -> GradTensor {
            try input.relu()
        }
        
        /// Apply GELU activation
        public static func gelu(_ input: GradTensor) throws -> GradTensor {
            try input.gelu()
        }
        
        /// Apply Sigmoid activation
        public static func sigmoid(_ input: GradTensor) throws -> GradTensor {
            let data = input.data.toArray()
            var result = try Tensor(shape: input.shape, dtype: input.data.dtype)
            let ptr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
            
            for i in 0..<input.count {
                ptr[i] = 1.0 / (1.0 + exp(-data[i]))
            }
            
            return GradTensor(result, requiresGrad: input.requiresGrad)
        }
        
        /// Apply Tanh activation
        public static func tanh(_ input: GradTensor) throws -> GradTensor {
            let data = input.data.toArray()
            var result = try Tensor(shape: input.shape, dtype: input.data.dtype)
            let ptr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
            
            for i in 0..<input.count {
                ptr[i] = Foundation.tanh(data[i])
            }
            
            return GradTensor(result, requiresGrad: input.requiresGrad)
        }
        
        /// Apply Softmax
        public static func softmax(_ input: GradTensor, dim: Int = -1) throws -> GradTensor {
            let data = input.data.toArray()
            let lastDim = input.shape.last!
            let numGroups = input.count / lastDim
            
            var result = try Tensor(shape: input.shape, dtype: input.data.dtype)
            let ptr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
            
            for g in 0..<numGroups {
                let offset = g * lastDim
                
                // Find max for stability
                var maxVal: Float = -Float.infinity
                for i in 0..<lastDim {
                    maxVal = max(maxVal, data[offset + i])
                }
                
                // Compute exp and sum
                var sumExp: Float = 0
                for i in 0..<lastDim {
                    ptr[offset + i] = exp(data[offset + i] - maxVal)
                    sumExp += ptr[offset + i]
                }
                
                // Normalize
                for i in 0..<lastDim {
                    ptr[offset + i] /= sumExp
                }
            }
            
            return GradTensor(result, requiresGrad: input.requiresGrad)
        }
        
        // MARK: - Loss Functions
        
        /// Cross entropy loss
        public static func crossEntropy(_ input: GradTensor, _ target: Tensor) throws -> GradTensor {
            try CrossEntropyLoss().forward(input, target)
        }
        
        /// Mean squared error loss
        public static func mse(_ input: GradTensor, _ target: Tensor) throws -> GradTensor {
            try MSELoss().forward(input, target)
        }
        
        /// Binary cross entropy loss
        public static func bce(_ input: GradTensor, _ target: Tensor) throws -> GradTensor {
            try BCELoss().forward(input, target)
        }
        
        /// L1 loss
        public static func l1(_ input: GradTensor, _ target: Tensor) throws -> GradTensor {
            try L1Loss().forward(input, target)
        }
        
        // MARK: - Linear Operations
        
        /// Linear transformation
        public static func linear(_ input: GradTensor, _ weight: GradTensor, _ bias: GradTensor? = nil) throws -> GradTensor {
            var output = try GradTensor.matmul(input, weight)
            if let bias = bias {
                output = try GradTensor.add(output, bias)
            }
            return output
        }
    }
}

// MARK: - Fusion.nn.init

extension Fusion.nn {
    
    /// Weight initialization methods
    public enum `init` {
        
        /// Xavier/Glorot uniform initialization
        public static func xavier(_ tensor: GradTensor) {
            let fanIn = tensor.shape.count > 1 ? tensor.shape[0] : 1
            let fanOut = tensor.shape.count > 1 ? tensor.shape[1] : tensor.shape[0]
            let scale = sqrt(6.0 / Float(fanIn + fanOut))
            
            let ptr = tensor.data.buffer.pointer.bindMemory(to: Float.self, capacity: tensor.count)
            for i in 0..<tensor.count {
                ptr[i] = Float.random(in: -scale...scale)
            }
        }
        
        /// Kaiming/He initialization for ReLU
        public static func kaiming(_ tensor: GradTensor) {
            let fanIn = tensor.shape.count > 1 ? tensor.shape[0] : 1
            let scale = sqrt(2.0 / Float(fanIn))
            
            let ptr = tensor.data.buffer.pointer.bindMemory(to: Float.self, capacity: tensor.count)
            for i in 0..<tensor.count {
                ptr[i] = Float.random(in: -1...1) * scale
            }
        }
        
        /// Zero initialization
        public static func zeros(_ tensor: GradTensor) {
            let ptr = tensor.data.buffer.pointer.bindMemory(to: Float.self, capacity: tensor.count)
            for i in 0..<tensor.count {
                ptr[i] = 0
            }
        }
        
        /// Ones initialization
        public static func ones(_ tensor: GradTensor) {
            let ptr = tensor.data.buffer.pointer.bindMemory(to: Float.self, capacity: tensor.count)
            for i in 0..<tensor.count {
                ptr[i] = 1
            }
        }
        
        /// Normal initialization
        public static func normal(_ tensor: GradTensor, mean: Float = 0, std: Float = 1) {
            let ptr = tensor.data.buffer.pointer.bindMemory(to: Float.self, capacity: tensor.count)
            for i in 0..<tensor.count {
                // Box-Muller transform for normal distribution
                let u1 = Float.random(in: 0.0001...1)
                let u2 = Float.random(in: 0...1)
                let z = sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
                ptr[i] = mean + std * z
            }
        }
    }
}
