// Fusion.linalg - Linear Algebra Module
// Matrix operations with intelligent routing

import Foundation
import Accelerate

extension Fusion {
    
    /// Linear algebra module - uses intelligent GPU+CPU routing
    public enum linalg {
        
        // MARK: - Matrix Operations
        
        /// Matrix multiplication (intelligent GPU+CPU split)
        @inlinable
        public static func matmul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
            try IntelligentRouter.shared.matmul(a, b)
        }
        
        /// Matrix multiplication for GradTensors
        @inlinable
        public static func matmul(_ a: GradTensor, _ b: GradTensor) throws -> GradTensor {
            try GradTensor.matmul(a, b)
        }
        
        /// Matrix transpose
        public static func transpose(_ a: Tensor) throws -> Tensor {
            try Tensor.transpose(a)
        }
        
        /// Matrix-vector multiplication
        public static func mv(_ a: Tensor, _ v: Tensor) throws -> Tensor {
            guard a.ndim == 2 && v.ndim == 1 else {
                throw LinalgError.dimensionMismatch
            }
            let vReshaped = try v.reshape([v.count, 1])
            let result = try matmul(a, vReshaped)
            return try result.reshape([a.shape[0]])
        }
        
        /// Vector dot product
        public static func dot(_ a: Tensor, _ b: Tensor) throws -> Float {
            guard a.ndim == 1 && b.ndim == 1 && a.count == b.count else {
                throw LinalgError.dimensionMismatch
            }
            
            let aData = a.toArray()
            let bData = b.toArray()
            var result: Float = 0
            vDSP_dotpr(aData, 1, bData, 1, &result, vDSP_Length(a.count))
            return result
        }
        
        /// Matrix norm (Frobenius)
        public static func norm(_ a: Tensor) throws -> Float {
            let data = a.toArray()
            var result: Float = 0
            vDSP_svesq(data, 1, &result, vDSP_Length(a.count))
            return sqrt(result)
        }
        
        /// Vector norm (L2)
        public static func vectorNorm(_ a: Tensor, ord: Int = 2) throws -> Float {
            let data = a.toArray()
            
            switch ord {
            case 1:
                return data.reduce(0) { $0 + abs($1) }
            case 2:
                var result: Float = 0
                vDSP_svesq(data, 1, &result, vDSP_Length(a.count))
                return sqrt(result)
            default:
                let p = Float(ord)
                return pow(data.reduce(0) { $0 + pow(abs($1), p) }, 1/p)
            }
        }
        
        // MARK: - Decompositions (simplified)
        
        /// Compute determinant of square matrix
        public static func det(_ a: Tensor) throws -> Float {
            guard a.ndim == 2 && a.shape[0] == a.shape[1] else {
                throw LinalgError.notSquare
            }
            
            // Simple 2x2 and 3x3 cases
            let n = a.shape[0]
            let data = a.toArray()
            
            if n == 2 {
                return data[0] * data[3] - data[1] * data[2]
            }
            
            // For larger matrices, use LU decomposition (simplified)
            // This is a placeholder - real impl would use LAPACK
            throw LinalgError.notImplemented("det for n > 2")
        }
        
        /// Matrix trace
        public static func trace(_ a: Tensor) throws -> Float {
            guard a.ndim == 2 && a.shape[0] == a.shape[1] else {
                throw LinalgError.notSquare
            }
            
            let data = a.toArray()
            let n = a.shape[0]
            var sum: Float = 0
            for i in 0..<n {
                sum += data[i * n + i]
            }
            return sum
        }
    }
}

// MARK: - Errors

public enum LinalgError: Error {
    case dimensionMismatch
    case notSquare
    case notImplemented(String)
}
