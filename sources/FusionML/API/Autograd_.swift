// Fusion.autograd - Automatic Differentiation Module
// Gradient computation and control

import Foundation

extension Fusion {
    
    /// Automatic differentiation module
    public enum autograd {
        
        // MARK: - Types
        
        /// GradTensor type for tensors with gradients
        public typealias Variable = GradTensor
        
        // MARK: - Gradient Computation
        
        /// Compute gradients of output with respect to inputs
        public static func grad(
            _ outputs: GradTensor,
            _ inputs: [GradTensor],
            gradOutputs: Tensor? = nil,
            retainGraph: Bool = false,
            createGraph: Bool = false
        ) throws {
            try outputs.backward(gradOutputs)
        }
        
        /// Backward pass (alias)
        public static func backward(_ output: GradTensor, gradient: Tensor? = nil) throws {
            try output.backward(gradient)
        }
        
        // MARK: - Context Managers
        
        /// Execute block without gradient tracking
        public static func noGrad<T>(_ block: () throws -> T) rethrows -> T {
            // In Swift we don't have context managers, but this signals intent
            // The block should use .detach() on tensors if needed
            try block()
        }
        
        /// Enable gradient mode (default)
        public static func enableGrad<T>(_ block: () throws -> T) rethrows -> T {
            try block()
        }
        
        // MARK: - Utility
        
        /// Create tensor with gradient tracking
        public static func variable(_ data: Tensor, requiresGrad: Bool = true) -> GradTensor {
            GradTensor(data, requiresGrad: requiresGrad)
        }
        
        /// Detach tensor from computation graph
        public static func detach(_ tensor: GradTensor) -> GradTensor {
            tensor.detach()
        }
        
        /// Zero gradients for all tensors
        public static func zeroGrad(_ tensors: [GradTensor]) {
            for t in tensors {
                t.zeroGrad()
            }
        }
    }
}
