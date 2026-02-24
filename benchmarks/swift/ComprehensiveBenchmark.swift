// Swift Benchmark - Comprehensive performance test
// Run with: swift run BenchmarkExample

import Foundation
import FusionML

@main
struct ComprehensiveBenchmark {
    static func main() throws {
        print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
        print("🔥 FusionML Swift Comprehensive Benchmark")
        print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
        
        Fusion.initialize()
        
        var results: [String: Any] = [:]
        
        // 1. MatMul Benchmarks
        print("\n📊 Matrix Multiplication:")
        let sizes = [256, 512, 1024, 2048, 4096]
        var matmulResults: [String: Double] = [:]
        
        for size in sizes {
            let time = try benchmarkMatmul(size: size)
            matmulResults["matmul_\(size)"] = time
            print("   \(size)x\(size): \(String(format: "%.2f", time)) ms")
        }
        results["matmul"] = matmulResults
        
        // 2. Backend Comparison
        print("\n📊 Backend Comparison (1024x1024):")
        let a = try Fusion.rand([1024, 1024])
        let b = try Fusion.rand([1024, 1024])
        
        // CPU
        var start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 { _ = try Fusion.cpu.matmul(a, b) }
        let cpuTime = (CFAbsoluteTimeGetCurrent() - start) / 5 * 1000
        print("   CPU:   \(String(format: "%.2f", cpuTime)) ms")
        
        // GPU
        start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 { _ = try Fusion.gpu.matmul(a, b) }
        let gpuTime = (CFAbsoluteTimeGetCurrent() - start) / 5 * 1000
        print("   GPU:   \(String(format: "%.2f", gpuTime)) ms")
        
        // Smart (GPU+CPU)
        start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 { _ = try Fusion.linalg.matmul(a, b) }
        let smartTime = (CFAbsoluteTimeGetCurrent() - start) / 5 * 1000
        print("   Smart: \(String(format: "%.2f", smartTime)) ms ⚡")
        
        let bestSingle = min(cpuTime, gpuTime)
        let speedup = ((bestSingle - smartTime) / bestSingle) * 100
        print("   Speedup: \(String(format: "+%.0f", speedup))%")
        
        results["backend_comparison"] = [
            "cpu_ms": cpuTime,
            "gpu_ms": gpuTime,
            "smart_ms": smartTime,
            "speedup_percent": speedup
        ]
        
        // 3. Training Benchmark
        print("\n📊 Training Benchmark:")
        let trainingTime = try benchmarkTraining()
        print("   MLP Training: \(String(format: "%.2f", trainingTime)) ms/epoch")
        results["training_ms_per_epoch"] = trainingTime
        
        // 4. Device Info
        results["device"] = Fusion.device
        results["date"] = ISO8601DateFormatter().string(from: Date())
        
        print("\n" + "=" .padding(toLength: 60, withPad: "=", startingAt: 0))
        print("✅ Benchmark Complete!")
        print("   Device: \(Fusion.device)")
        print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
    }
    
    static func benchmarkMatmul(size: Int) throws -> Double {
        let a = try Fusion.rand([size, size])
        let b = try Fusion.rand([size, size])
        
        // Warmup
        for _ in 0..<3 { _ = try Fusion.linalg.matmul(a, b) }
        
        // Benchmark
        let iterations = 10
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = try Fusion.linalg.matmul(a, b)
        }
        return (CFAbsoluteTimeGetCurrent() - start) / Double(iterations) * 1000
    }
    
    static func benchmarkTraining() throws -> Double {
        let model = Fusion.nn.sequential(
            try Fusion.nn.linear(784, 256),
            Fusion.nn.relu(),
            try Fusion.nn.linear(256, 10)
        )
        
        let optimizer = Fusion.optim.adam(model.parameters(), lr: 0.01)
        let batchSize = 32
        let numBatches = 50
        
        // Warmup
        for _ in 0..<5 {
            let x = GradTensor(try Fusion.rand([batchSize, 784]), requiresGrad: true)
            let target = try Fusion.random.randint(0, 10, [batchSize])
            let output = try model.forward(x)
            let loss = try Fusion.nn.functional.crossEntropy(output, target)
            try Fusion.autograd.backward(loss)
            try optimizer.step()
            optimizer.zeroGrad()
        }
        
        // Benchmark
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<numBatches {
            let x = GradTensor(try Fusion.rand([batchSize, 784]), requiresGrad: true)
            let target = try Fusion.random.randint(0, 10, [batchSize])
            let output = try model.forward(x)
            let loss = try Fusion.nn.functional.crossEntropy(output, target)
            try Fusion.autograd.backward(loss)
            try optimizer.step()
            optimizer.zeroGrad()
        }
        
        return (CFAbsoluteTimeGetCurrent() - start) / Double(numBatches) * 1000
    }
}
