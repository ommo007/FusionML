// SmartScheduler3Way - GPU + ANE + CPU Proportional Splitting
// Uses all three compute units for maximum throughput

import Foundation
import CoreML
import Accelerate
import Metal


/// Three-way smart scheduler using GPU, ANE, and CPU
public final class SmartScheduler3: @unchecked Sendable {
    
    public static let shared = SmartScheduler3()
    
    // Performance profiles
    private var profiles: [String: [HardwareBackend: PerformanceProfile]] = [:]
    private let lock = NSLock()
    
    // Core ML models for ANE
    private var aneModels: [Int: MLModel] = [:]
    private let modelLock = NSLock()
    
    private init() {}
    
    // MARK: - Model Loading
    
    private func loadANEModel(size: Int) throws -> MLModel {
        modelLock.lock()
        if let cached = aneModels[size] {
            modelLock.unlock()
            return cached
        }
        modelLock.unlock()
        
        // Use absolute path to models directory
        let basePath = "/Users/ommohite/Documents/Programming/FusionML/models"
        let mlpackagePath = "\(basePath)/matmul_\(size).mlpackage"
        let mlpackageURL = URL(fileURLWithPath: mlpackagePath)
        
        // Compile the model first
        let compiledURL = try MLModel.compileModel(at: mlpackageURL)
        
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine  // Prefer ANE
        
        let model = try MLModel(contentsOf: compiledURL, configuration: config)
        
        modelLock.lock()
        aneModels[size] = model
        modelLock.unlock()
        
        return model
    }
    
    // MARK: - ANE Matmul
    
    private func aneMatmul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        let size = a.shape[0]
        let model = try loadANEModel(size: size)
        
        let M = a.shape[0]
        let K = a.shape[1]
        let N = b.shape[1]
        
        // Create MLMultiArrays
        let aArray = try MLMultiArray(shape: [M, K] as [NSNumber], dataType: .float16)
        let bArray = try MLMultiArray(shape: [K, N] as [NSNumber], dataType: .float16)
        
        // Convert Float to Float16 during copy
        aArray.withUnsafeMutableBytes { ptr, strides in
            let dest = ptr.baseAddress!.assumingMemoryBound(to: Float16.self)
            let src = a.buffer.pointer.assumingMemoryBound(to: Float.self)
            for i in 0..<(M * K) {
                dest[i] = Float16(src[i])
            }
        }
        bArray.withUnsafeMutableBytes { ptr, strides in
            let dest = ptr.baseAddress!.assumingMemoryBound(to: Float16.self)
            let src = b.buffer.pointer.assumingMemoryBound(to: Float.self)
            for i in 0..<(K * N) {
                dest[i] = Float16(src[i])
            }
        }
        
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "a": MLFeatureValue(multiArray: aArray),
            "b": MLFeatureValue(multiArray: bArray)
        ])
        
        let output = try model.prediction(from: input)
        
        guard let resultArray = output.featureValue(for: "var_3")?.multiArrayValue else {
            throw ANEError.invalidInput
        }
        
        let result = try Tensor(shape: [M, N], dtype: .float32)
        let ptr = result.buffer.pointer.bindMemory(to: Float.self, capacity: M * N)
        
        // Convert Float16 back to Float
        resultArray.withUnsafeBytes { ptrBuffer in
            let src = ptrBuffer.baseAddress!.assumingMemoryBound(to: Float16.self)
            for i in 0..<(M * N) {
                ptr[i] = Float(src[i])
            }
        }
        
        return result
    }
    
    // MARK: - Calibration
    
    public func calibrate(sizes: [Int] = [512, 1024, 2048]) throws {
        print("📐 Calibrating 3-Way SmartScheduler (GPU + ANE + CPU)...")
        
        for size in sizes {
            let a = try Tensor.random([size, size])
            let b = try Tensor.random([size, size])
            let key = "matmul_\(size)"
            
            // CPU
            let cpuTime = try measureBackend {
                try Tensor.matmul(a, b)
            }
            recordProfile(key: key, backend: .cpu, timeMs: cpuTime)
            let cpuGFLOPS = (2.0 * Double(size * size * size) / cpuTime) / 1_000_000
            
            // GPU
            let gpuTime = try measureBackend {
                try GPUEngine.shared.matmulMPS(a, b)
            }
            recordProfile(key: key, backend: .gpu, timeMs: gpuTime)
            let gpuGFLOPS = (2.0 * Double(size * size * size) / gpuTime) / 1_000_000
            
            // ANE
            var aneGFLOPS = 0.0
            do {
                let aneTime = try measureBackend {
                    try self.aneMatmul(a, b)
                }
                recordProfile(key: key, backend: .ane, timeMs: aneTime)
                aneGFLOPS = (2.0 * Double(size * size * size) / aneTime) / 1_000_000
            } catch {
                let msg = error.localizedDescription
                print("  ⚠️ ANE error for \(size)×\(size): \(msg)")
            }
            
            print("  \(size)×\(size): CPU \(String(format: "%.0f", cpuGFLOPS)), GPU \(String(format: "%.0f", gpuGFLOPS)), ANE \(String(format: "%.0f", aneGFLOPS)) GFLOPS")
        }
        
        print("✅ Calibration complete!")
    }
    
    private func measureBackend<T>(_ operation: () throws -> T) throws -> Double {
        // Warmup
        for _ in 0..<3 { _ = try operation() }
        
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 { _ = try operation() }
        return (CFAbsoluteTimeGetCurrent() - start) * 1000 / 5
    }
    
    private func recordProfile(key: String, backend: HardwareBackend, timeMs: Double) {
        lock.lock()
        defer { lock.unlock() }
        
        if profiles[key] == nil { profiles[key] = [:] }
        if profiles[key]![backend] == nil { profiles[key]![backend] = PerformanceProfile() }
        profiles[key]![backend]!.record(timeMs)
    }
    
    // MARK: - 3-Way Split Ratio
    
    public func optimalSplitRatio(for size: Int) -> (cpu: Double, gpu: Double, ane: Double) {
        let key = "matmul_\(size)"
        
        lock.lock()
        let profs = profiles[key]
        lock.unlock()
        
        guard let profs = profs else {
            return (0.34, 0.33, 0.33)
        }
        
        let cpuThroughput = profs[.cpu]?.throughput ?? 0
        let gpuThroughput = profs[.gpu]?.throughput ?? 0
        let aneThroughput = profs[.ane]?.throughput ?? 0
        
        let total = cpuThroughput + gpuThroughput + aneThroughput
        guard total > 0 else { return (0.34, 0.33, 0.33) }
        
        return (cpuThroughput / total, gpuThroughput / total, aneThroughput / total)
    }
    
    // MARK: - 3-Way Smart Matmul
    
    public func smartMatmul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        let M = a.shape[0]
        let N = b.shape[1]
        
        // For small matrices, use single backend
        if M < 512 {
            return try Tensor.matmul(a, b)
        }
        
        // Get split ratios
        let (cpuRatio, gpuRatio, aneRatio) = optimalSplitRatio(for: M)
        
        // If one backend dominates (>80%), just use it
        if cpuRatio > 0.8 { return try Tensor.matmul(a, b) }
        if gpuRatio > 0.8 { return try GPUEngine.shared.matmulMPS(a, b) }
        if aneRatio > 0.8, aneModels[M] != nil { return try aneMatmul(a, b) }
        
        // Calculate row splits
        let cpuRows = Int(Double(M) * cpuRatio)
        let gpuRows = Int(Double(M) * gpuRatio)
        let aneRows = M - cpuRows - gpuRows
        
        let result = try Tensor(shape: [M, N], dtype: a.dtype)
        let group = DispatchGroup()
        let threadErrors = ThreadSafeErrors()
        
        // Bind pointers and extract buffers on the caller thread
        let aPtr = a.buffer.pointer.bindMemory(to: Float.self, capacity: a.count)
        let bPtr = b.buffer.pointer.bindMemory(to: Float.self, capacity: b.count)
        let rPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: result.count)
        
        let aMetal = a.metalBuffer
        let bMetal = b.metalBuffer
        let rMetal = result.metalBuffer
        
        let K = a.shape[1]

        // GPU portion (rows cpuRows..<cpuRows+gpuRows)
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
                    threadErrors.append(error)
                }
                group.leave()
            }
        }

        // ANE portion (rows cpuRows+gpuRows..<M)
        if aneRows > 0 {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async { [aPtr, b, rPtr] in
                do {
                    let startRow = cpuRows + gpuRows
                    if self.aneModels[M] != nil {
                        // Create padded input A of shape [M, K]
                        let paddedA = try Tensor.zeros([M, K])
                        let paddedAPtr = paddedA.buffer.pointer.bindMemory(to: Float.self, capacity: paddedA.count)
                        
                        // Copy the ANE slice rows into the top of paddedA
                        memcpy(paddedAPtr, aPtr.advanced(by: startRow * K), aneRows * K * 4)
                        
                        // Run ANE prediction
                        let ANEPaddedResult = try self.aneMatmul(paddedA, b)
                        let ANEPaddedResultPtr = ANEPaddedResult.buffer.pointer.bindMemory(to: Float.self, capacity: ANEPaddedResult.count)
                        
                        // Copy the computed rows back to the result matrix
                        memcpy(rPtr.advanced(by: startRow * N), ANEPaddedResultPtr, aneRows * N * 4)
                    } else {
                        // Fallback to CPU matmul for slice
                        let aSlice = try Tensor(shape: [aneRows, K], dtype: .float32)
                        let aSlicePtr = aSlice.buffer.pointer.bindMemory(to: Float.self, capacity: aSlice.count)
                        memcpy(aSlicePtr, aPtr.advanced(by: startRow * K), aneRows * K * 4)
                        
                        let tempResult = try Tensor.matmul(aSlice, b)
                        let tempResultPtr = tempResult.buffer.pointer.bindMemory(to: Float.self, capacity: tempResult.count)
                        memcpy(rPtr.advanced(by: startRow * N), tempResultPtr, aneRows * N * 4)
                    }
                } catch {
                    threadErrors.append(error)
                }
                group.leave()
            }
        }
        
        // CPU portion (rows 0..<cpuRows) - runs synchronously in parallel with GPU/ANE
        if cpuRows > 0 {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       Int32(cpuRows), Int32(N), Int32(K), 1.0,
                       aPtr, Int32(K), bPtr, Int32(N), 0.0, rPtr, Int32(N))
        }
        
        group.wait()
        
        try threadErrors.check()
        
        return result
    }
    
    private func sliceRows(_ tensor: Tensor, start: Int, count: Int) throws -> Tensor {
        let K = tensor.shape[1]
        let result = try Tensor(shape: [count, K], dtype: tensor.dtype)
        memcpy(result.buffer.pointer,
               tensor.buffer.pointer.advanced(by: start * K * 4),
               count * K * 4)
        return result
    }
    
    private func copyRows(from src: Tensor, to dst: Tensor, startRow: Int) {
        let N = dst.shape[1]
        let rowCount = src.shape[0]
        memcpy(dst.buffer.pointer.advanced(by: startRow * N * 4),
               src.buffer.pointer,
               rowCount * N * 4)
    }
    
    // MARK: - Stats
    
    public func printStats() {
        lock.lock()
        defer { lock.unlock() }
        
        print("\n📊 3-Way SmartScheduler Profiles:")
        for (key, backends) in profiles.sorted(by: { $0.key < $1.key }) {
            print("\n\(key):")
            let total = backends.values.reduce(0.0) { $0 + $1.throughput }
            for (backend, profile) in backends.sorted(by: { $0.1.avgTimeMs < $1.1.avgTimeMs }) {
                let pct = total > 0 ? (profile.throughput / total) * 100 : 0
                print("  \(backend.rawValue): \(String(format: "%.2f", profile.avgTimeMs)) ms → \(String(format: "%.0f", pct))% of work")
            }
        }
    }
}

/// Thread-safe array of errors for concurrent execution paths
private final class ThreadSafeErrors: @unchecked Sendable {
    private var errors: [Error] = []
    private let lock = NSLock()
    
    func append(_ error: Error) {
        lock.lock()
        errors.append(error)
        lock.unlock()
    }
    
    func check() throws {
        lock.lock()
        defer { lock.unlock() }
        if let first = errors.first {
            throw first
        }
    }
}
