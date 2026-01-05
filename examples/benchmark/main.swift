// BenchmarkExample - Performance comparison across backends
// Shows intelligent routing advantage

import Foundation
import FusionML

@main
struct BenchmarkExample {
    static func main() throws {
        print("🔥 FusionML Benchmark")
        print("=" .padding(toLength: 60, withPad: "=", startingAt: 0))
        
        Fusion.initialize()
        
        let sizes = [512, 1024, 2048]
        
        for size in sizes {
            print("\n📊 Matrix Size: \(size)x\(size)")
            print("-" .padding(toLength: 40, withPad: "-", startingAt: 0))
            
            let a = try Fusion.rand([size, size])
            let b = try Fusion.rand([size, size])
            
            // Warmup
            _ = try Fusion.cpu.matmul(a, b)
            _ = try Fusion.gpu.matmul(a, b)
            _ = try Fusion.linalg.matmul(a, b)
            
            // CPU benchmark
            var start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<3 {
                _ = try Fusion.cpu.matmul(a, b)
            }
            let cpuTime = (CFAbsoluteTimeGetCurrent() - start) / 3 * 1000
            
            // GPU benchmark
            start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<3 {
                _ = try Fusion.gpu.matmul(a, b)
            }
            let gpuTime = (CFAbsoluteTimeGetCurrent() - start) / 3 * 1000
            
            // Smart routing benchmark
            start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<3 {
                _ = try Fusion.linalg.matmul(a, b)
            }
            let smartTime = (CFAbsoluteTimeGetCurrent() - start) / 3 * 1000
            
            let bestSingle = min(cpuTime, gpuTime)
            let speedup = ((bestSingle - smartTime) / bestSingle) * 100
            
            print("   CPU:   \(String(format: "%6.2f", cpuTime)) ms")
            print("   GPU:   \(String(format: "%6.2f", gpuTime)) ms")
            print("   Smart: \(String(format: "%6.2f", smartTime)) ms ⚡")
            if speedup > 0 {
                print("   Speedup: +\(String(format: "%.0f", speedup))%")
            }
        }
        
        print("\n" + "=" .padding(toLength: 60, withPad: "=", startingAt: 0))
        print("✅ Benchmark Complete!")
        print("   Smart routing uses GPU+CPU parallel execution")
    }
}
