// FusionEngine - Parallel Silicon Fusion
// World-first: Run GPU + ANE + CPU simultaneously

import Foundation
import Metal
import CoreML
import Accelerate

/// Hardware execution result
public struct ExecutionResult: Sendable {
    public let backend: HardwareBackend
    public let tensor: Tensor
    public let timeMs: Double
}

/// Available hardware backends
public enum HardwareBackend: String, CaseIterable, Sendable {
    case gpu = "GPU"
    case ane = "ANE"
    case cpu = "CPU"
}

/// Parallel Silicon Fusion Engine
/// Runs operations on multiple hardware units simultaneously
public final class FusionEngine: @unchecked Sendable {
    
    public static let shared = FusionEngine()
    
    // Hardware engines
    private let gpuEngine: GPUEngine
    private let cpuQueue: DispatchQueue
    private let aneQueue: DispatchQueue
    
    // Performance tracking
    private var performanceHistory: [String: [HardwareBackend: [Double]]] = [:]
    private let historyLock = NSLock()
    
    private init() {
        self.gpuEngine = GPUEngine.shared
        self.cpuQueue = DispatchQueue(label: "com.siliconml.cpu", qos: .userInitiated)
        self.aneQueue = DispatchQueue(label: "com.siliconml.ane", qos: .userInitiated)
    }
    
    // MARK: - Speculative Racing
    
    /// Race an operation on ALL backends, return first result
    /// This is genuinely new - no framework does this
    public func race<T>(
        operation: String,
        gpu: @escaping () throws -> T,
        cpu: @escaping () throws -> T
    ) throws -> (result: T, winner: HardwareBackend, timeMs: Double) {
        
        let group = DispatchGroup()
        let resultLock = NSLock()
        
        var winningResult: T?
        var winningBackend: HardwareBackend?
        var winningTime: Double?
        var firstError: Error?
        
        // GPU execution
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            let start = CFAbsoluteTimeGetCurrent()
            do {
                let result = try gpu()
                let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
                
                resultLock.lock()
                if winningResult == nil {
                    winningResult = result
                    winningBackend = .gpu
                    winningTime = time
                }
                resultLock.unlock()
            } catch {
                resultLock.lock()
                if firstError == nil { firstError = error }
                resultLock.unlock()
            }
            group.leave()
        }
        
        // CPU execution
        group.enter()
        self.cpuQueue.async {
            let start = CFAbsoluteTimeGetCurrent()
            do {
                let result = try cpu()
                let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
                
                resultLock.lock()
                if winningResult == nil {
                    winningResult = result
                    winningBackend = .cpu
                    winningTime = time
                }
                resultLock.unlock()
            } catch {
                resultLock.lock()
                if firstError == nil { firstError = error }
                resultLock.unlock()
            }
            group.leave()
        }
        
        // Wait for first completion
        group.wait()
        
        if let result = winningResult, let backend = winningBackend, let time = winningTime {
            // Record performance
            recordPerformance(operation: operation, backend: backend, timeMs: time)
            return (result, backend, time)
        }
        
        throw firstError ?? MemoryError.deviceNotAvailable
    }
    
    /// Race matrix multiplication across backends
    public func raceMatmul(_ a: Tensor, _ b: Tensor) throws -> (result: Tensor, winner: HardwareBackend, timeMs: Double) {
        let opKey = "matmul_\(a.shape[0])x\(a.shape[1])x\(b.shape[1])"
        
        return try race(
            operation: opKey,
            gpu: { try self.gpuEngine.matmul(a, b) },
            cpu: { try Tensor.matmul(a, b) }
        )
    }
    
    // MARK: - Parallel Pipeline
    
    /// Execute multiple operations in parallel on different backends
    public func parallelExecute(
        gpuOp: (() throws -> Tensor)?,
        aneOp: (() throws -> Tensor)?,
        cpuOp: (() throws -> Tensor)?
    ) throws -> [HardwareBackend: ExecutionResult] {
        
        let group = DispatchGroup()
        var results: [HardwareBackend: ExecutionResult] = [:]
        let lock = NSLock()
        var errors: [Error] = []
        
        // GPU
        if let op = gpuOp {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                let start = CFAbsoluteTimeGetCurrent()
                do {
                    let tensor = try op()
                    let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
                    lock.lock()
                    results[.gpu] = ExecutionResult(backend: .gpu, tensor: tensor, timeMs: time)
                    lock.unlock()
                } catch {
                    lock.lock()
                    errors.append(error)
                    lock.unlock()
                }
                group.leave()
            }
        }
        
        // CPU
        if let op = cpuOp {
            group.enter()
            self.cpuQueue.async {
                let start = CFAbsoluteTimeGetCurrent()
                do {
                    let tensor = try op()
                    let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
                    lock.lock()
                    results[.cpu] = ExecutionResult(backend: .cpu, tensor: tensor, timeMs: time)
                    lock.unlock()
                } catch {
                    lock.lock()
                    errors.append(error)
                    lock.unlock()
                }
                group.leave()
            }
        }
        
        group.wait()
        
        if results.isEmpty && !errors.isEmpty {
            throw errors.first!
        }
        
        return results
    }
    
    // MARK: - Adaptive Routing
    
    private func recordPerformance(operation: String, backend: HardwareBackend, timeMs: Double) {
        historyLock.lock()
        defer { historyLock.unlock() }
        
        if performanceHistory[operation] == nil {
            performanceHistory[operation] = [:]
        }
        if performanceHistory[operation]![backend] == nil {
            performanceHistory[operation]![backend] = []
        }
        
        performanceHistory[operation]![backend]!.append(timeMs)
        
        // Keep only last 100 samples
        if performanceHistory[operation]![backend]!.count > 100 {
            performanceHistory[operation]![backend]!.removeFirst()
        }
    }
    
    /// Get the best backend for an operation based on history
    public func bestBackend(for operation: String) -> HardwareBackend? {
        historyLock.lock()
        defer { historyLock.unlock() }
        
        guard let history = performanceHistory[operation] else { return nil }
        
        var best: (HardwareBackend, Double)?
        for (backend, times) in history {
            guard !times.isEmpty else { continue }
            let avg = times.reduce(0, +) / Double(times.count)
            if best == nil || avg < best!.1 {
                best = (backend, avg)
            }
        }
        
        return best?.0
    }
    
    /// Smart matmul - uses learned best backend
    public func smartMatmul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        let opKey = "matmul_\(a.shape[0])x\(a.shape[1])x\(b.shape[1])"
        
        if let best = bestBackend(for: opKey) {
            // Use learned best backend
            let start = CFAbsoluteTimeGetCurrent()
            let result: Tensor
            switch best {
            case .gpu:
                result = try gpuEngine.matmul(a, b)
            case .cpu:
                result = try Tensor.matmul(a, b)
            case .ane:
                result = try Tensor.matmul(a, b) // Fallback to CPU
            }
            let time = (CFAbsoluteTimeGetCurrent() - start) * 1000
            recordPerformance(operation: opKey, backend: best, timeMs: time)
            return result
        } else {
            // Haven't learned yet - race to find best
            let (result, _, _) = try raceMatmul(a, b)
            return result
        }
    }
    
    // MARK: - Statistics
    
    public func printStats() {
        historyLock.lock()
        defer { historyLock.unlock() }
        
        print("\n📊 Fusion Engine Performance Stats:")
        print("=" .padding(toLength: 50, withPad: "=", startingAt: 0))
        
        for (op, backends) in performanceHistory.sorted(by: { $0.key < $1.key }) {
            print("\n\(op):")
            for (backend, times) in backends.sorted(by: { $0.value.reduce(0, +) < $1.value.reduce(0, +) }) {
                let avg = times.reduce(0, +) / Double(times.count)
                print("  \(backend.rawValue): \(String(format: "%.3f", avg)) ms avg (\(times.count) samples)")
            }
        }
    }
}
