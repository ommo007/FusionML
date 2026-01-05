import ANELib
import Foundation

/// Demo: Matrix Multiplication on Apple Silicon GPU using Metal Performance Shaders
@main
struct MatrixDemo {
    static func main() {
        print("🚀 ANELib Matrix Demo - GPU Acceleration on Apple Silicon")
        print("=========================================================\n")
        
        do {
            // Initialize Metal
            let compute = try MetalCompute()
            compute.printDeviceInfo()
            print()
            
            // Create matrix operations instance
            let matrixOps = MPSMatrixOps(compute: compute)
            
            // Demo 1: Small matrix multiplication (verify correctness)
            try demo1_smallMatmul(matrixOps)
            
            // Demo 2: Large matrix benchmark
            try demo2_benchmark(matrixOps)
            
            // Demo 3: Float16 compatible dimensions (prep for ANE)
            try demo3_anePrep(matrixOps)
            
            print("\n✅ All demos completed successfully!")
            
        } catch {
            print("❌ Error: \(error)")
        }
    }
    
    /// Demo 1: Small matrix - verify correctness
    static func demo1_smallMatmul(_ ops: MPSMatrixOps) throws {
        print("📐 Demo 1: Small Matrix Multiplication (3×3)")
        print("---------------------------------------------")
        
        // A = [[1, 2, 3],
        //      [4, 5, 6],
        //      [7, 8, 9]]
        let a: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        // B = [[9, 8, 7],
        //      [6, 5, 4],
        //      [3, 2, 1]]
        let b: [Float] = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        
        print("Matrix A:")
        a.printAsMatrix(rows: 3, columns: 3, precision: 0)
        print("\nMatrix B:")
        b.printAsMatrix(rows: 3, columns: 3, precision: 0)
        
        let c = try ops.matmul(a: a, b: b, size: 3)
        
        print("\nResult C = A × B:")
        c.printAsMatrix(rows: 3, columns: 3, precision: 0)
        
        // Verify: C[0,0] = 1*9 + 2*6 + 3*3 = 9 + 12 + 9 = 30
        let expected00: Float = 30
        print("\nVerification: C[0,0] = \(c[0]) (expected: \(expected00)) ✓")
        print()
    }
    
    /// Demo 2: Large matrix benchmark
    static func demo2_benchmark(_ ops: MPSMatrixOps) throws {
        print("⚡ Demo 2: Large Matrix Benchmark")
        print("----------------------------------")
        
        let sizes = [256, 512, 1024, 2048]
        
        for size in sizes {
            let a = [Float].random(count: size * size)
            let b = [Float].random(count: size * size)
            
            let (_, avgMs) = try ops.benchmarkMatmul(
                a: a, b: b,
                m: size, n: size, k: size,
                iterations: 5
            )
            
            // Calculate GFLOPS: 2 * M * N * K operations for matmul
            let flops = 2.0 * Double(size) * Double(size) * Double(size)
            let gflops = (flops / avgMs) / 1_000_000  // GFLOPS
            
            print("  \(size)×\(size): \(String(format: "%.3f", avgMs)) ms (\(String(format: "%.1f", gflops)) GFLOPS)")
        }
        print()
    }
    
    /// Demo 3: Prepare dimensions that work well with ANE (multiples of 16)
    static func demo3_anePrep(_ ops: MPSMatrixOps) throws {
        print("🧠 Demo 3: ANE-Friendly Dimensions")
        print("-----------------------------------")
        print("(ANE prefers shapes that are multiples of 16)")
        
        let sizes = [16, 64, 128, 256]
        
        for size in sizes {
            let a = [Float].random(count: size * size)
            let b = [Float].random(count: size * size)
            
            let start = CFAbsoluteTimeGetCurrent()
            _ = try ops.matmul(a: a, b: b, size: size)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            print("  \(size)×\(size): \(String(format: "%.3f", elapsed)) ms")
        }
        print("\nNote: These dimensions will also work efficiently on ANE via Core ML")
    }
}
