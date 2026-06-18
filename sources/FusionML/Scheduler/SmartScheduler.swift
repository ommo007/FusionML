// SmartScheduler - Intelligent Proportional Work Distribution
// Profiles hardware, learns performance, splits work optimally

import Foundation
import Accelerate
import Metal


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
    // Empirically tuned split ratios: [operation_size -> cpuRatio]
    private var tunedRatios: [String: Double] = [:]
    private let lock = NSLock()
    
    // Calibration status
    private var isCalibrated = false
    
    private init() {}
    
    // MARK: - Calibration
    
    /// Calibrate by measuring actual backend performance and sweeping concurrent split ratios
    public func calibrate(sizes: [Int] = [256, 512, 1024, 2048]) throws {
        print("📐 Calibrating SmartScheduler...")
        
        for size in sizes {
            try autoreleasepool {
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
                
                // Localized search sweep around the theoretical ratio to maximize efficiency
                let cpuThroughput = 1.0 / cpuTime
                let gpuThroughput = 1.0 / gpuTime
                let total = cpuThroughput + gpuThroughput
                
                let theoreticalRatio = cpuThroughput / total
                var bestRatio = theoreticalRatio
                var minTime = Double.infinity
                
                let startRatio = max(0.05, theoreticalRatio - 0.15)
                let endRatio = min(0.95, theoreticalRatio + 0.15)
                
                // Fine-grained localized sweep to find the physical optimum
                for r in stride(from: startRatio, through: endRatio, by: 0.03) {
                    autoreleasepool {
                        do {
                            let testTime = try measureSplit(a, b, cpuRatio: r)
                            if testTime < minTime {
                                minTime = testTime
                                bestRatio = r
                            }
                        } catch {
                            // Ignore
                        }
                    }
                }
                
                lock.lock()
                tunedRatios[key] = bestRatio
                lock.unlock()
                
                let optimalGFLOPS = (2.0 * Double(size * size * size) / minTime) / 1_000_000
                print("  \(size)×\(size): CPU \(String(format: "%.0f", cpuGFLOPS)) GFLOPS, GPU \(String(format: "%.0f", gpuGFLOPS)) GFLOPS, Smart Split (Tuned Ratio: \(String(format: "%.2f", bestRatio))) \(String(format: "%.0f", optimalGFLOPS)) GFLOPS")
                
                // Clean up calibration buffers immediately
                MemoryManager.shared.clearPool()
            }
        }
        
        isCalibrated = true
        print("✅ Calibration complete!")
    }
    
    /// Calibrate shapes by measuring actual performance and sweeping split ratios
    public func calibrateShapes(_ shapes: [(M: Int, N: Int, K: Int)]) throws {
        print("📐 Calibrating SmartScheduler Shapes...")
        for shape in shapes {
            try autoreleasepool {
                let M = shape.M
                let N = shape.N
                let K = shape.K
                let key = "matmul_\(M)_\(N)_\(K)"
                
                let a = try Tensor.random([M, K])
                let b = try Tensor.random([K, N])
                
                // Measure CPU
                let cpuTime = try measureBackend(.cpu, size: M) {
                    try Tensor.matmul(a, b)
                }
                recordProfile(key: key, backend: .cpu, timeMs: cpuTime)
                
                // Measure GPU
                let gpuTime = try measureBackend(.gpu, size: M) {
                    try GPUEngine.shared.matmulMPS(a, b)
                }
                recordProfile(key: key, backend: .gpu, timeMs: gpuTime)
                
                let cpuThroughput = 1.0 / cpuTime
                let gpuThroughput = 1.0 / gpuTime
                let total = cpuThroughput + gpuThroughput
                let theoreticalRatio = cpuThroughput / total
                
                var bestRatio = theoreticalRatio
                var minTime = Double.infinity
                
                let startRatio = max(0.05, theoreticalRatio - 0.15)
                let endRatio = min(0.95, theoreticalRatio + 0.15)
                
                for r in stride(from: startRatio, through: endRatio, by: 0.03) {
                    autoreleasepool {
                        do {
                            let testTime = try measureSplit(a, b, cpuRatio: r)
                            if testTime < minTime {
                                minTime = testTime
                                bestRatio = r
                            }
                        } catch {}
                    }
                }
                
                lock.lock()
                tunedRatios[key] = bestRatio
                lock.unlock()
                
                let cpuGFLOPS = (2.0 * Double(M * N * K) / cpuTime) / 1_000_000
                let gpuGFLOPS = (2.0 * Double(M * N * K) / gpuTime) / 1_000_000
                let optimalGFLOPS = (2.0 * Double(M * N * K) / minTime) / 1_000_000
                print("  \(M)×\(N)×\(K): CPU \(String(format: "%.0f", cpuGFLOPS)) GFLOPS, GPU \(String(format: "%.0f", gpuGFLOPS)) GFLOPS, Smart Split (Tuned Ratio: \(String(format: "%.2f", bestRatio))) \(String(format: "%.0f", optimalGFLOPS)) GFLOPS")
                
                MemoryManager.shared.clearPool()
            }
        }
        isCalibrated = true
        print("✅ Shapes calibration complete!")
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
    
    private func measureSplit(_ a: Tensor, _ b: Tensor, cpuRatio: Double) throws -> Double {
        // Warmup
        for _ in 0..<2 {
            try autoreleasepool {
                _ = try smartMatmulWithRatio(a, b, cpuRatio: cpuRatio)
            }
        }
        
        // Measure
        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 3
        for _ in 0..<iterations {
            try autoreleasepool {
                _ = try smartMatmulWithRatio(a, b, cpuRatio: cpuRatio)
            }
        }
        return (CFAbsoluteTimeGetCurrent() - start) * 1000 / Double(iterations)
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
    public func optimalSplitRatio(for M: Int, N: Int, K: Int) -> (cpu: Double, gpu: Double) {
        let key = "matmul_\(M)_\(N)_\(K)"
        
        lock.lock()
        let tuned = tunedRatios[key]
        lock.unlock()
        
        if let cpuRatio = tuned {
            return (cpuRatio, 1.0 - cpuRatio)
        }
        
        // Fallback to square calibration
        lock.lock()
        let squareTuned = tunedRatios["matmul_\(M)"]
        lock.unlock()
        if let cpuRatio = squareTuned {
            return (cpuRatio, 1.0 - cpuRatio)
        }
        
        // Default
        return (0.30, 0.70)
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
        let K = a.shape[1]
        
        // For small matrices, use single best backend
        if M < 512 {
            return try Tensor.matmul(a, b)  // CPU is usually best for small
        }
        
        // Get optimal split ratio
        let (cpuRatio, _) = optimalSplitRatio(for: M, N: N, K: K)
        return try smartMatmulWithRatio(a, b, cpuRatio: cpuRatio)
    }
    
    private func smartMatmulWithRatio(_ a: Tensor, _ b: Tensor, cpuRatio: Double) throws -> Tensor {
        let M = a.shape[0]
        let N = b.shape[1]
        let K = a.shape[1]
        
        let cpuRows = Int(Double(M) * cpuRatio)
        let gpuRows = M - cpuRows
        
        // If one backend dominates, just use it
        if cpuRatio > 0.95 {
            return try Tensor.matmul(a, b)
        }
        if cpuRatio < 0.05 {
            return try GPUEngine.shared.matmulMPS(a, b)
        }
        
        let result = try Tensor(shape: [M, N], dtype: a.dtype)
        let group = DispatchGroup()
        var gpuError: Error?
        
        // Bind pointers and extract buffers on the caller thread
        let aPtr = a.buffer.pointer.bindMemory(to: Float.self, capacity: a.count)
        let bPtr = b.buffer.pointer.bindMemory(to: Float.self, capacity: b.count)
        let rPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
        
        let aMetal = a.metalBuffer
        let bMetal = b.metalBuffer
        let rMetal = result.metalBuffer
        
        // GPU portion
        if gpuRows > 0 {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async { [aMetal, bMetal, rMetal] in
                do {
                    try GPUEngine.shared.matmulRawMPS(
                        a: aMetal,
                        b: bMetal,
                        result: rMetal,
                        M: gpuRows,
                        N: N,
                        K: K,
                        aOffset: cpuRows * K * 4,
                        resultOffset: cpuRows * N * 4
                    )
                } catch {
                    gpuError = error
                }
                group.leave()
            }
        }
        
        // CPU portion (runs synchronously on caller thread in parallel with GPU)
        if cpuRows > 0 {
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
        }
        
        group.wait()
        
        if let error = gpuError {
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
