// Fusion - Main Namespace
// PyTorch-style API: Fusion.nn, Fusion.optim, Fusion.linalg, etc.

import Foundation

/// Main namespace for SiliconML framework
/// Access functionality via: Fusion.nn, Fusion.optim, Fusion.linalg, etc.
public enum Fusion {
    
    /// Framework version
    public static let version = "0.2.0"
    
    /// Initialize the framework
    public static func initialize() {
        SiliconML.initialize()
    }
    
    // MARK: - Submodules (defined in separate files)
    // Fusion.nn       → NN.swift
    // Fusion.optim    → Optim_.swift
    // Fusion.autograd → Autograd_.swift
    // Fusion.linalg   → Linalg.swift
    // Fusion.cpu      → CPU.swift
    // Fusion.gpu      → GPU.swift
    // Fusion.ane      → ANE.swift
    // Fusion.data     → Data_.swift
    // Fusion.random   → Random.swift
}

// MARK: - Tensor Creation (top-level convenience)

extension Fusion {
    
    /// Create tensor from array
    public static func tensor(_ data: [Float], shape: [Int]? = nil) throws -> Tensor {
        try Tensor(data, shape: shape)
    }
    
    /// Create zeros tensor
    public static func zeros(_ shape: [Int]) throws -> Tensor {
        try Tensor.zeros(shape)
    }
    
    /// Create ones tensor
    public static func ones(_ shape: [Int]) throws -> Tensor {
        try Tensor.ones(shape)
    }
    
    /// Create random tensor
    public static func rand(_ shape: [Int]) throws -> Tensor {
        try Tensor.random(shape)
    }
    
    /// Create identity matrix
    public static func eye(_ size: Int) throws -> Tensor {
        try Tensor.eye(size)
    }
    
    /// Create tensor requiring gradients
    public static func tensor(_ data: [Float], shape: [Int]? = nil, requiresGrad: Bool) throws -> GradTensor {
        let t = try Tensor(data, shape: shape)
        return GradTensor(t, requiresGrad: requiresGrad)
    }
}

// MARK: - Device Info

extension Fusion {
    
    /// Get current device name
    public static var device: String {
        GPUEngine.shared.device.name
    }
    
    /// Check if GPU is available
    public static var isGPUAvailable: Bool {
        true  // Always available on Apple Fusion
    }
    
    /// Check if ANE is available  
    public static var isANEAvailable: Bool {
        true  // Always available on Apple Fusion
    }
    
    /// Memory statistics
    public static var memoryStats: (allocated: Int, reused: Int) {
        let stats = MemoryManager.shared.stats
        return (stats.allocated, stats.reused)
    }
}
