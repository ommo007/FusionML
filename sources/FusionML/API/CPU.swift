// Fusion.cpu - CPU Backend Module
// Direct CPU operations using Accelerate/AMX

import Foundation
import Accelerate

extension Fusion {
    
    /// CPU backend - direct access to Accelerate/AMX operations
    public enum cpu {
        
        // MARK: - Matrix Operations
        
        /// Matrix multiplication on CPU (uses AMX on Apple Fusion)
        public static func matmul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
            try Tensor.matmul(a, b)
        }
        
        /// Element-wise addition on CPU
        public static func add(_ a: Tensor, _ b: Tensor) throws -> Tensor {
            try Tensor.add(a, b)
        }
        
        /// Element-wise multiplication on CPU
        public static func mul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
            try Tensor.mul(a, b)
        }
        
        // MARK: - Reductions
        
        /// Sum all elements
        public static func sum(_ a: Tensor) -> Float {
            a.toArray().reduce(0, +)
        }
        
        /// Mean of all elements
        public static func mean(_ a: Tensor) -> Float {
            sum(a) / Float(a.count)
        }
        
        /// Max element
        public static func max(_ a: Tensor) -> Float {
            a.toArray().max() ?? 0
        }
        
        /// Min element
        public static func min(_ a: Tensor) -> Float {
            a.toArray().min() ?? 0
        }
        
        // MARK: - Element-wise Operations
        
        /// Apply exp
        public static func exp(_ a: Tensor) throws -> Tensor {
            let data = a.toArray()
            var result = try Tensor(shape: a.shape, dtype: a.dtype)
            let ptr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
            
            var count = Int32(a.count)
            vvexpf(ptr, data, &count)
            
            return result
        }
        
        /// Apply log
        public static func log(_ a: Tensor) throws -> Tensor {
            let data = a.toArray()
            var result = try Tensor(shape: a.shape, dtype: a.dtype)
            let ptr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
            
            var count = Int32(a.count)
            vvlogf(ptr, data, &count)
            
            return result
        }
        
        /// Apply sqrt
        public static func sqrt(_ a: Tensor) throws -> Tensor {
            let data = a.toArray()
            var result = try Tensor(shape: a.shape, dtype: a.dtype)
            let ptr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
            
            var count = Int32(a.count)
            vvsqrtf(ptr, data, &count)
            
            return result
        }
        
        /// Apply abs
        public static func abs(_ a: Tensor) throws -> Tensor {
            let data = a.toArray()
            var result = try Tensor(shape: a.shape, dtype: a.dtype)
            let ptr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
            
            vDSP_vabs(data, 1, ptr, 1, vDSP_Length(a.count))
            
            return result
        }
    }
}
