// Fusion.gpu - GPU Backend Module  
// Direct GPU operations using Metal/MPS

import Foundation

extension Fusion {
    
    /// GPU backend - direct access to Metal/MPS operations
    public enum gpu {
        
        // MARK: - Matrix Operations
        
        /// Matrix multiplication on GPU (uses MPS)
        public static func matmul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
            try GPUEngine.shared.matmulMPS(a, b)
        }
        
        /// Element-wise addition on GPU
        public static func add(_ a: Tensor, _ b: Tensor) throws -> Tensor {
            try GPUEngine.shared.add(a, b)
        }
        
        /// ReLU on GPU
        public static func relu(_ a: Tensor) throws -> Tensor {
            try GPUEngine.shared.relu(a)
        }
        
        /// GELU on GPU
        public static func gelu(_ a: Tensor) throws -> Tensor {
            try GPUEngine.shared.gelu(a)
        }
        
        // MARK: - Device Info
        
        /// GPU device name
        public static var deviceName: String {
            GPUEngine.shared.device.name
        }
        
        /// Check if GPU is available
        public static var isAvailable: Bool {
            true  // Always available on Apple Fusion
        }
        
        /// Synchronize GPU (wait for all operations to complete)
        public static func synchronize() {
            // In Metal, we use waitUntilCompleted on command buffers
            // This is a hint to the user - actual sync happens per operation
        }
    }
}
