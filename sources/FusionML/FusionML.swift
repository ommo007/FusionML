// SiliconML - Ultra-Fast Apple Silicon ML Library
// Leverages GPU + ANE + CPU for maximum performance

import Metal
import Accelerate

/// SiliconML main module
/// High-performance ML library for Apple Silicon
public struct SiliconML {
    
    /// Library version
    public static let version = "0.1.0"
    
    /// Initialize the library
    public static func initialize() {
        _ = MemoryManager.shared
        _ = GPUEngine.shared
        print("🚀 SiliconML \(version) initialized")
        print("   Device: \(MemoryManager.shared.device.name)")
    }
    
    /// Get memory statistics
    public static var memoryStats: (allocated: Int, reused: Int, pooled: Int, active: Int) {
        MemoryManager.shared.stats
    }
}

// MARK: - Operator Overloads for Tensor

extension Tensor {
    
    /// Element-wise add
    public static func + (lhs: Tensor, rhs: Tensor) throws -> Tensor {
        try GPUEngine.shared.add(lhs, rhs)
    }
    
    /// Element-wise multiply
    public static func * (lhs: Tensor, rhs: Tensor) throws -> Tensor {
        try GPUEngine.shared.mul(lhs, rhs)
    }
}

// MARK: - Functional API

/// Matrix multiplication
public func matmul(_ a: Tensor, _ b: Tensor, backend: Backend = .auto) throws -> Tensor {
    switch backend {
    case .gpu:
        return try GPUEngine.shared.matmul(a, b)
    case .mps:
        return try GPUEngine.shared.matmulMPS(a, b)
    case .cpu:
        return try Tensor.matmul(a, b)
    case .auto:
        // Use GPU for large matrices, CPU for small
        if a.count > 10000 {
            return try GPUEngine.shared.matmul(a, b)
        } else {
            return try Tensor.matmul(a, b)
        }
    }
}

/// Element-wise addition
public func add(_ a: Tensor, _ b: Tensor, backend: Backend = .auto) throws -> Tensor {
    switch backend {
    case .gpu, .mps, .auto:
        return try GPUEngine.shared.add(a, b)
    case .cpu:
        return try Tensor.add(a, b)
    }
}

/// ReLU activation
public func relu(_ x: Tensor) throws -> Tensor {
    try GPUEngine.shared.relu(x)
}

/// GELU activation
public func gelu(_ x: Tensor) throws -> Tensor {
    try GPUEngine.shared.gelu(x)
}

/// Compute backend
public enum Backend {
    case auto  // Automatically choose best
    case gpu   // Custom Metal shaders
    case mps   // Metal Performance Shaders
    case cpu   // Accelerate/BLAS
}

// MARK: - Benchmarking

/// Benchmark a tensor operation
public func benchmark<T>(_ name: String, iterations: Int = 100, warmup: Int = 10, _ operation: () throws -> T) rethrows -> (result: T, timeMs: Double, gflops: Double?) {
    // Warmup
    for _ in 0..<warmup {
        _ = try operation()
    }
    
    // Benchmark
    let start = CFAbsoluteTimeGetCurrent()
    var result: T!
    for _ in 0..<iterations {
        result = try operation()
    }
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000 // ms
    let avgTime = elapsed / Double(iterations)
    
    print("\(name): \(String(format: "%.3f", avgTime)) ms")
    
    return (result, avgTime, nil)
}

/// Benchmark matrix multiplication with GFLOPS calculation
public func benchmarkMatmul(size: Int, backend: Backend = .auto, iterations: Int = 100) throws -> (timeMs: Double, gflops: Double) {
    let a = try Tensor.random([size, size])
    let b = try Tensor.random([size, size])
    
    // Warmup
    for _ in 0..<10 {
        _ = try matmul(a, b, backend: backend)
    }
    
    // Benchmark
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = try matmul(a, b, backend: backend)
    }
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000 // ms
    let avgTime = elapsed / Double(iterations)
    
    // Calculate GFLOPS: 2 * N^3 / time
    let flops = 2.0 * Double(size * size * size)
    let gflops = (flops / avgTime) / 1_000_000
    
    return (avgTime, gflops)
}
