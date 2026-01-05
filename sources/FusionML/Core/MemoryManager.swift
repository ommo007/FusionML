// SiliconML - Unified Memory Manager
// Zero-copy buffer management for Apple Silicon unified memory architecture

import Metal
import Foundation

/// Errors related to memory management
public enum MemoryError: Error, CustomStringConvertible {
    case deviceNotAvailable
    case allocationFailed(size: Int)
    case bufferNotFound(id: UUID)
    case invalidShape
    
    public var description: String {
        switch self {
        case .deviceNotAvailable:
            return "Metal device not available"
        case .allocationFailed(let size):
            return "Failed to allocate \(size) bytes"
        case .bufferNotFound(let id):
            return "Buffer \(id) not found"
        case .invalidShape:
            return "Invalid tensor shape"
        }
    }
}

/// Data types supported by SiliconML
public enum DType: Sendable {
    case float32
    case float16
    case int32
    case int64
    
    public var size: Int {
        switch self {
        case .float32: return 4
        case .float16: return 2
        case .int32: return 4
        case .int64: return 8
        }
    }
    
    public var metalType: MTLPixelFormat {
        switch self {
        case .float32: return .r32Float
        case .float16: return .r16Float
        case .int32: return .r32Sint
        case .int64: return .r32Sint // Approximation
        }
    }
}

/// Unified memory buffer wrapper
/// Provides zero-copy access across CPU, GPU, and ANE
public final class UnifiedBuffer: @unchecked Sendable {
    public let id: UUID
    public let size: Int
    public let buffer: MTLBuffer
    public let device: MTLDevice
    
    /// Direct pointer to unified memory (accessible from CPU)
    public var pointer: UnsafeMutableRawPointer {
        buffer.contents()
    }
    
    init(device: MTLDevice, size: Int, options: MTLResourceOptions = .storageModeShared) throws {
        guard let buffer = device.makeBuffer(length: size, options: options) else {
            throw MemoryError.allocationFailed(size: size)
        }
        self.id = UUID()
        self.size = size
        self.buffer = buffer
        self.device = device
    }
    
    /// Create buffer from existing data (copies data once)
    init<T>(device: MTLDevice, data: [T]) throws {
        let size = data.count * MemoryLayout<T>.stride
        guard let buffer = device.makeBuffer(bytes: data, length: size, options: .storageModeShared) else {
            throw MemoryError.allocationFailed(size: size)
        }
        self.id = UUID()
        self.size = size
        self.buffer = buffer
        self.device = device
    }
    
    /// Read data as typed array
    public func read<T>(as type: T.Type, count: Int) -> [T] {
        let ptr = pointer.bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
    
    /// Write data to buffer
    public func write<T>(_ data: [T]) {
        let ptr = pointer.bindMemory(to: T.self, capacity: data.count)
        for (i, value) in data.enumerated() {
            ptr[i] = value
        }
    }
}

/// Memory pool for efficient buffer reuse
public final class MemoryManager: @unchecked Sendable {
    public static let shared = MemoryManager()
    
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    
    private var bufferPool: [Int: [UnifiedBuffer]] = [:]  // Size -> Available buffers
    private var activeBuffers: [UUID: UnifiedBuffer] = [:]
    private let lock = NSLock()
    
    // Statistics
    public private(set) var totalAllocated: Int = 0
    public private(set) var totalReused: Int = 0
    
    private init() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            fatalError("Metal not available on this device")
        }
        self.device = device
        self.commandQueue = queue
    }
    
    /// Allocate a buffer (reuses from pool if possible)
    public func allocate(size: Int) throws -> UnifiedBuffer {
        lock.lock()
        defer { lock.unlock() }
        
        // Round up to power of 2 for better pooling
        let alignedSize = nextPowerOf2(size)
        
        // Try to reuse from pool
        if var available = bufferPool[alignedSize], !available.isEmpty {
            let buffer = available.removeLast()
            bufferPool[alignedSize] = available
            activeBuffers[buffer.id] = buffer
            totalReused += 1
            return buffer
        }
        
        // Allocate new buffer
        let buffer = try UnifiedBuffer(device: device, size: alignedSize)
        activeBuffers[buffer.id] = buffer
        totalAllocated += 1
        return buffer
    }
    
    /// Allocate buffer initialized with data
    public func allocate<T>(from data: [T]) throws -> UnifiedBuffer {
        lock.lock()
        defer { lock.unlock() }
        
        let buffer = try UnifiedBuffer(device: device, data: data)
        activeBuffers[buffer.id] = buffer
        totalAllocated += 1
        return buffer
    }
    
    /// Release buffer back to pool for reuse
    public func release(_ buffer: UnifiedBuffer) {
        lock.lock()
        defer { lock.unlock() }
        
        activeBuffers.removeValue(forKey: buffer.id)
        
        if bufferPool[buffer.size] == nil {
            bufferPool[buffer.size] = []
        }
        bufferPool[buffer.size]?.append(buffer)
    }
    
    /// Clear all pooled buffers (keeps active buffers)
    public func clearPool() {
        lock.lock()
        defer { lock.unlock() }
        bufferPool.removeAll()
    }
    
    /// Get memory statistics
    public var stats: (allocated: Int, reused: Int, pooled: Int, active: Int) {
        lock.lock()
        defer { lock.unlock() }
        let pooledCount = bufferPool.values.reduce(0) { $0 + $1.count }
        return (totalAllocated, totalReused, pooledCount, activeBuffers.count)
    }
    
    private func nextPowerOf2(_ n: Int) -> Int {
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        return v + 1
    }
}
