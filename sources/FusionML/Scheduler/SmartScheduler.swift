// SmartScheduler - Intelligent Proportional Work Distribution
// Profiles hardware, learns performance, splits work optimally

import Foundation
import Accelerate

/// Measured performance for a backend at a specific workload
public struct PerformanceProfile: Codable {
    public var samples: [Double] = []  // Time in ms
    public var totalCount: Int = 0
    
    public var avgTimeMs: Double {
        guard !samples.isEmpty else { return Double.infinity }
        return samples.reduce(0, +) / Double(samples.count)
    }
    
    public var throughput: Double {  // ops per ms
        guard avgTimeMs > 0 else { return 0 }
        return 1.0 / avgTimeMs
    }
    
    mutating func record(_ timeMs: Double) {
        samples.append(timeMs)
        totalCount += 1
        // Keep rolling window of last 20 samples
        if samples.count > 20 {
            samples.removeFirst()
        }
    }
}

/// Smart scheduler that learns and adapts
public final class SmartScheduler: @unchecked Sendable {
    
    public static let shared = SmartScheduler()
    
    // Performance profiles: [operation_size -> [backend -> profile]]
    private var profiles: [String: [HardwareBackend: PerformanceProfile]] = [:]
    private let lock = NSLock()
    
    // Calibration status
    private var isCalibrated = false
    
    private init() {}
    
    // MARK: - Calibration
    
    /// Calibrate by measuring actual backend performance
    public func calibrate(sizes: [Int] = [256, 512, 1024, 2048]) throws {
        print("📐 Calibrating SmartScheduler...")
        
        for size in sizes {
            let a = try Tensor.random([size, size])
            let b = try Tensor.random([size, size])
            
            let key = "matmul_\(size)"
            
            // Measure CPU
            let cpuTime = try measureBackend(.cpu, size: size) {
                try Tensor.matmul(a, b)
            }
            recordProfile(key: key, backend: .cpu, timeMs: cpuTime)
            
            // Measure GPU (MPS)
            let gpuTime = try measureBackend(.gpu, size: size) {
                try GPUEngine.shared.matmulMPS(a, b)
            }
            recordProfile(key: key, backend: .gpu, timeMs: gpuTime)
            
            let cpuGFLOPS = (2.0 * Double(size * size * size) / cpuTime) / 1_000_000
            let gpuGFLOPS = (2.0 * Double(size * size * size) / gpuTime) / 1_000_000
            print("  \(size)×\(size): CPU \(String(format: "%.0f", cpuGFLOPS)) GFLOPS, GPU \(String(format: "%.0f", gpuGFLOPS)) GFLOPS")
        }
        
        isCalibrated = true
        print("✅ Calibration complete!")
    }
    
    private func measureBackend<T>(_ backend: HardwareBackend, size: Int, operation: () throws -> T) throws -> Double {
        // Warmup
        for _ in 0..<3 {
            _ = try operation()
        }
        
        // Measure
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 {
            _ = try operation()
        }
        return (CFAbsoluteTimeGetCurrent() - start) * 1000 / 5
    }
    
    private func recordProfile(key: String, backend: HardwareBackend, timeMs: Double) {
        lock.lock()
        defer { lock.unlock() }
        
        if profiles[key] == nil {
            profiles[key] = [:]
        }
        if profiles[key]![backend] == nil {
            profiles[key]![backend] = PerformanceProfile()
        }
        profiles[key]![backend]!.record(timeMs)
    }
    
    // MARK: - Optimal Split Ratios
    
    /// Calculate optimal work split ratio based on measured throughput
    public func optimalSplitRatio(for size: Int) -> (cpu: Double, gpu: Double) {
        let key = "matmul_\(size)"
        
        lock.lock()
        let profs = profiles[key]
        lock.unlock()
        
        guard let profs = profs,
              let cpuProf = profs[.cpu],
              let gpuProf = profs[.gpu] else {
            // Default 50/50 if not calibrated
            return (0.5, 0.5)
        }
        
        let cpuThroughput = cpuProf.throughput
        let gpuThroughput = gpuProf.throughput
        let total = cpuThroughput + gpuThroughput
        
        guard total > 0 else { return (0.5, 0.5) }
        
        return (cpuThroughput / total, gpuThroughput / total)
    }
    
    // MARK: - Intelligent Matmul
    
    /// Matrix multiply with intelligent proportional splitting
    public func smartMatmul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        guard a.ndim == 2 && b.ndim == 2 else {
            throw MemoryError.invalidShape
        }
        guard a.shape[1] == b.shape[0] else {
            throw MemoryError.invalidShape
        }
        
        let M = a.shape[0]
        let N = b.shape[1]
        
        // For small matrices, use single best backend
        if M < 512 {
            return try Tensor.matmul(a, b)  // CPU is usually best for small
        }
        
        // Get optimal split ratio
        let (cpuRatio, gpuRatio) = optimalSplitRatio(for: M)
        
        // Calculate row splits
        let cpuRows = Int(Double(M) * cpuRatio)
        let gpuRows = M - cpuRows
        
        // If one backend dominates, just use it
        if cpuRatio > 0.9 {
            return try Tensor.matmul(a, b)
        }
        if gpuRatio > 0.9 {
            return try GPUEngine.shared.matmulMPS(a, b)
        }
        
        // Parallel split execution
        let result = try Tensor(shape: [M, N], dtype: a.dtype)
        let group = DispatchGroup()
        var cpuError: Error?
        var gpuError: Error?
        
        // CPU portion
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // Direct pointer arithmetic for zero-copy view
                let aPtr = a.buffer.pointer.bindMemory(to: Float.self, capacity: a.count)
                let bPtr = b.buffer.pointer.bindMemory(to: Float.self, capacity: b.count)
                let rPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
                
                let K = a.shape[1]
                
                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    Int32(cpuRows),
                    Int32(N),
                    Int32(K),
                    1.0,
                    aPtr, Int32(K),  // First cpuRows rows of A
                    bPtr, Int32(N),
                    0.0,
                    rPtr, Int32(N)   // First cpuRows rows of result
                )
            } catch {
                cpuError = error
            }
            group.leave()
        }
        
        // GPU portion
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // Create view of remaining rows
                let aSlice = try self.tensorView(a, rowStart: cpuRows, rowCount: gpuRows)
                let gpuResult = try GPUEngine.shared.matmulMPS(aSlice, b)
                
                // Copy to result
                let srcPtr = gpuResult.buffer.pointer
                let dstPtr = result.buffer.pointer.advanced(by: cpuRows * N * 4)
                memcpy(dstPtr, srcPtr, gpuRows * N * 4)
            } catch {
                gpuError = error
            }
            group.leave()
        }
        
        group.wait()
        
        if let error = cpuError ?? gpuError {
            throw error
        }
        
        return result
    }
    
    /// Create a view (copy) of tensor rows
    private func tensorView(_ tensor: Tensor, rowStart: Int, rowCount: Int) throws -> Tensor {
        let K = tensor.shape[1]
        let view = try Tensor(shape: [rowCount, K], dtype: tensor.dtype)
        
        let srcPtr = tensor.buffer.pointer.advanced(by: rowStart * K * tensor.dtype.size)
        memcpy(view.buffer.pointer, srcPtr, rowCount * K * tensor.dtype.size)
        
        return view
    }
    
    // MARK: - Statistics
    
    public func printStats() {
        lock.lock()
        defer { lock.unlock() }
        
        print("\n📊 SmartScheduler Performance Profiles:")
        print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
        
        for (key, backends) in profiles.sorted(by: { $0.key < $1.key }) {
            print("\n\(key):")
            
            var throughputs: [(HardwareBackend, Double)] = []
            for (backend, profile) in backends {
                throughputs.append((backend, profile.throughput))
                print("  \(backend.rawValue): \(String(format: "%.2f", profile.avgTimeMs)) ms avg")
            }
            
            // Show optimal split
            let total = throughputs.reduce(0) { $0 + $1.1 }
            if total > 0 {
                for (backend, throughput) in throughputs {
                    let pct = (throughput / total) * 100
                    print("    → \(backend.rawValue) should get \(String(format: "%.0f", pct))% of work")
                }
            }
        }
    }
}
