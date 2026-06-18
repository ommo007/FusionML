// BenchmarkExample - Comprehensive performance comparison across backends
// Evaluates both raw matrix operations and three major models (Llama-3-8B, GPT-2 XL, Deep MLP)

import Foundation
import FusionML

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public func print(_ items: Any..., separator: String = " ", terminator: String = "\n") {
    let output = items.map { "\($0)" }.joined(separator: separator)
    Swift.print(output, terminator: terminator)
    fflush(stdout)
}

// MARK: - Llama3 Transformer Block (approximated with GeGLU using gelu)
public final class Llama3Block: Module {
    public var training: Bool = true {
        didSet {
            qProj.training = training
            kProj.training = training
            vProj.training = training
            oProj.training = training
            ln1.training = training
            gateProj.training = training
            upProj.training = training
            downProj.training = training
            ln2.training = training
        }
    }
    
    let qProj: GradLinear
    let kProj: GradLinear
    let vProj: GradLinear
    let oProj: GradLinear
    let ln1: GradLayerNorm
    let gateProj: GradLinear
    let upProj: GradLinear
    let downProj: GradLinear
    let ln2: GradLayerNorm
    
    public init(dim: Int = 4096, hiddenDim: Int = 14336) throws {
        self.qProj = try GradLinear(inFeatures: dim, outFeatures: dim, bias: false)
        self.kProj = try GradLinear(inFeatures: dim, outFeatures: dim, bias: false)
        self.vProj = try GradLinear(inFeatures: dim, outFeatures: dim, bias: false)
        self.oProj = try GradLinear(inFeatures: dim, outFeatures: dim, bias: false)
        self.ln1 = try GradLayerNorm(normalizedShape: [dim])
        self.gateProj = try GradLinear(inFeatures: dim, outFeatures: hiddenDim, bias: false)
        self.upProj = try GradLinear(inFeatures: dim, outFeatures: hiddenDim, bias: false)
        self.downProj = try GradLinear(inFeatures: hiddenDim, outFeatures: dim, bias: false)
        self.ln2 = try GradLayerNorm(normalizedShape: [dim])
    }
    
    public func forward(_ input: GradTensor) throws -> GradTensor {
        // Pre-LN Self-Attention
        let norm1 = try ln1.forward(input)
        let q = try qProj.forward(norm1)
        let k = try kProj.forward(norm1)
        let v = try vProj.forward(norm1)
        
        let kT = try GradTensor(Tensor.transpose(k.data), requiresGrad: k.requiresGrad)
        let scores = try GradTensor.matmul(q, kT)
        let attended = try GradTensor.matmul(scores, v)
        let projected = try oProj.forward(attended)
        
        let h1 = try GradTensor.add(input, projected)
        
        // Pre-LN SwiGLU MLP (approximated with GeGLU using gelu)
        let norm2 = try ln2.forward(h1)
        let gate = try gateProj.forward(norm2)
        let up = try upProj.forward(norm2)
        
        let activatedGate = try gate.gelu()
        let intermediate = try GradTensor.mul(activatedGate, up)
        let output = try downProj.forward(intermediate)
        
        return try GradTensor.add(h1, output)
    }
    
    public func parameters() -> [GradTensor] {
        return qProj.parameters() + kProj.parameters() + vProj.parameters() + oProj.parameters() +
               ln1.parameters() + gateProj.parameters() + upProj.parameters() + downProj.parameters() +
               ln2.parameters()
    }
    
    public func namedParameters() -> [(String, GradTensor)] {
        return qProj.namedParameters() + kProj.namedParameters() + vProj.namedParameters() + oProj.namedParameters() +
               ln1.namedParameters() + gateProj.namedParameters() + upProj.namedParameters() + downProj.namedParameters() +
               ln2.namedParameters()
    }
}

// MARK: - GPT2 Transformer Block
public final class GPT2Block: Module {
    public var training: Bool = true {
        didSet {
            qProj.training = training
            kProj.training = training
            vProj.training = training
            oProj.training = training
            ln1.training = training
            ff1.training = training
            ff2.training = training
            ln2.training = training
        }
    }
    
    let qProj: GradLinear
    let kProj: GradLinear
    let vProj: GradLinear
    let oProj: GradLinear
    let ln1: GradLayerNorm
    let ff1: GradLinear
    let ff2: GradLinear
    let ln2: GradLayerNorm
    
    public init(dim: Int = 1600, ffnDim: Int = 6400) throws {
        self.qProj = try GradLinear(inFeatures: dim, outFeatures: dim, bias: true)
        self.kProj = try GradLinear(inFeatures: dim, outFeatures: dim, bias: true)
        self.vProj = try GradLinear(inFeatures: dim, outFeatures: dim, bias: true)
        self.oProj = try GradLinear(inFeatures: dim, outFeatures: dim, bias: true)
        self.ln1 = try GradLayerNorm(normalizedShape: [dim])
        self.ff1 = try GradLinear(inFeatures: dim, outFeatures: ffnDim, bias: true)
        self.ff2 = try GradLinear(inFeatures: ffnDim, outFeatures: dim, bias: true)
        self.ln2 = try GradLayerNorm(normalizedShape: [dim])
    }
    
    public func forward(_ input: GradTensor) throws -> GradTensor {
        // Pre-LN Self-Attention
        let norm1 = try ln1.forward(input)
        let q = try qProj.forward(norm1)
        let k = try kProj.forward(norm1)
        let v = try vProj.forward(norm1)
        
        let kT = try GradTensor(Tensor.transpose(k.data), requiresGrad: k.requiresGrad)
        let scores = try GradTensor.matmul(q, kT)
        let attended = try GradTensor.matmul(scores, v)
        let projected = try oProj.forward(attended)
        
        let h1 = try GradTensor.add(input, projected)
        
        // Pre-LN MLP
        let norm2 = try ln2.forward(h1)
        let intermediate = try ff1.forward(norm2)
        let activated = try intermediate.gelu()
        let output = try ff2.forward(activated)
        
        return try GradTensor.add(h1, output)
    }
    
    public func parameters() -> [GradTensor] {
        return qProj.parameters() + kProj.parameters() + vProj.parameters() + oProj.parameters() +
               ln1.parameters() + ff1.parameters() + ff2.parameters() + ln2.parameters()
    }
    
    public func namedParameters() -> [(String, GradTensor)] {
        return qProj.namedParameters() + kProj.namedParameters() + vProj.namedParameters() + oProj.namedParameters() +
               ln1.namedParameters() + ff1.namedParameters() + ff2.namedParameters() + ln2.namedParameters()
    }
}

// MARK: - Benchmark Execution
@main
struct BenchmarkExample {
    static func main() throws {
        print("🔥 FusionML Model Training & Inference Benchmarks")
        print("=" .padding(toLength: 65, withPad: "=", startingAt: 0))
        
        Fusion.initialize()
        
        // Calibrate once globally with dynamic split ratio optimization for actual model shapes
        let shapes = [
            (M: 1024, N: 4096, K: 4096),
            (M: 1024, N: 2048, K: 4096),
            (M: 1024, N: 4096, K: 2048),
            (M: 1024, N: 14336, K: 4096),
            (M: 1024, N: 4096, K: 14336),
            
            (M: 1024, N: 1600, K: 1600),
            (M: 1024, N: 2048, K: 1600),
            (M: 1024, N: 1600, K: 2048),
            (M: 1024, N: 6400, K: 1600),
            (M: 1024, N: 1600, K: 6400),
            
            (M: 1024, N: 10, K: 4096),
            
            // Training transposed backprop shapes
            (M: 4096, N: 4096, K: 1024),
            (M: 14336, N: 4096, K: 1024),
            (M: 6400, N: 1600, K: 1024)
        ]
        try SmartScheduler.shared.calibrateShapes(shapes)
        MemoryManager.shared.clearPool()
        
        // Calibrate 3-way split scheduler for square sizes supported on ANE
        try SmartScheduler3.shared.calibrate(sizes: [512, 1024, 2048])
        MemoryManager.shared.clearPool()
        
        let runEval = { (model: Module, x: GradTensor) throws -> Double in
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<3 {
                try autoreleasepool {
                    _ = try model.forward(x)
                }
            }
            return (CFAbsoluteTimeGetCurrent() - start) / 3 * 1000
        }
        
        let runTrainStep = { (model: Module, optimizer: Optimizer, x: GradTensor, y: Tensor) throws -> Double in
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<3 {
                try autoreleasepool {
                    optimizer.zeroGrad()
                    let output = try model.forward(x)
                    let loss = try Fusion.nn.functional.crossEntropy(output, y)
                    try Fusion.autograd.backward(loss)
                    try optimizer.step()
                }
            }
            return (CFAbsoluteTimeGetCurrent() - start) / 3 * 1000
        }
        
        let printTable = { (title: String, cpu: Double, gpu: Double, smart: Double) in
            let bestSingle = min(cpu, gpu)
            let speedup = ((bestSingle - smart) / bestSingle) * 100
            let ratio = bestSingle / smart
            
            print("\n📊 \(title):")
            print("   CPU-Only:    \(String(format: "%6.2f", cpu)) ms")
            print("   GPU-Only:    \(String(format: "%6.2f", gpu)) ms")
            print("   Smart Split:  \(String(format: "%6.2f", smart)) ms ⚡")
            if speedup > 0 {
                print("   Winner:       Smart Split (\(String(format: "%.1f", speedup))% latency reduction / \(String(format: "%.2f", ratio))x speedup)")
            } else {
                let winner = cpu < gpu ? "CPU" : "GPU"
                print("   Winner:       \(winner) (overhead bounds splitting)")
            }
        }
        
        func runLlama3() throws {
            print("\n📝 MODEL 1: Llama-3-8B Decoder Layer Block")
            print("   Batch: 4 | Tokens: 256 (SeqLen: 1024) | Hidden Dim: 4096 | FFN Dim: 14336")
            print("-" .padding(toLength: 65, withPad: "-", startingAt: 0))
            
            let llama = try Llama3Block()
            let llamaX = try Fusion.rand([1024, 4096])
            let llamaY = try Tensor(shape: [1024], dtype: .float32)
            let llamaInput = GradTensor(llamaX, requiresGrad: false)
            let llamaOpt = Fusion.optim.adam(llama.parameters(), lr: 0.01)
            
            // A. Inference
            llama.eval()
            IntelligentRouter.shared.forcedBackend = .cpu
            IntelligentRouter.shared.enableSplitting = false
            let llamaEvalCpu = try runEval(llama, llamaInput)
            
            IntelligentRouter.shared.forcedBackend = .gpu
            IntelligentRouter.shared.enableSplitting = false
            let llamaEvalGpu = try runEval(llama, llamaInput)
            
            IntelligentRouter.shared.forcedBackend = nil
            IntelligentRouter.shared.enableSplitting = true
            let llamaEvalSmart = try runEval(llama, llamaInput)
            printTable("Llama-3-8B Inference", llamaEvalCpu, llamaEvalGpu, llamaEvalSmart)
            
            // B. Training
            llama.train()
            IntelligentRouter.shared.forcedBackend = .cpu
            IntelligentRouter.shared.enableSplitting = false
            let llamaTrainCpu = try runTrainStep(llama, llamaOpt, llamaInput, llamaY)
            
            IntelligentRouter.shared.forcedBackend = .gpu
            IntelligentRouter.shared.enableSplitting = false
            let llamaTrainGpu = try runTrainStep(llama, llamaOpt, llamaInput, llamaY)
            
            IntelligentRouter.shared.forcedBackend = nil
            IntelligentRouter.shared.enableSplitting = true
            let llamaTrainSmart = try runTrainStep(llama, llamaOpt, llamaInput, llamaY)
            printTable("Llama-3-8B Training Step", llamaTrainCpu, llamaTrainGpu, llamaTrainSmart)
        }
        
        func runGPT2() throws {
            print("\n📝 MODEL 2: GPT-2 XL Decoder Layer Block")
            print("   Batch: 4 | Tokens: 256 (SeqLen: 1024) | Hidden Dim: 1600 | FFN Dim: 6400")
            print("-" .padding(toLength: 65, withPad: "-", startingAt: 0))
            
            let gpt2 = try GPT2Block()
            let gpt2X = try Fusion.rand([1024, 1600])
            let gpt2Y = try Tensor(shape: [1024], dtype: .float32)
            let gpt2Input = GradTensor(gpt2X, requiresGrad: false)
            let gpt2Opt = Fusion.optim.adam(gpt2.parameters(), lr: 0.01)
            
            // A. Inference
            gpt2.eval()
            IntelligentRouter.shared.forcedBackend = .cpu
            IntelligentRouter.shared.enableSplitting = false
            let gpt2EvalCpu = try runEval(gpt2, gpt2Input)
            
            IntelligentRouter.shared.forcedBackend = .gpu
            IntelligentRouter.shared.enableSplitting = false
            let gpt2EvalGpu = try runEval(gpt2, gpt2Input)
            
            IntelligentRouter.shared.forcedBackend = nil
            IntelligentRouter.shared.enableSplitting = true
            let gpt2EvalSmart = try runEval(gpt2, gpt2Input)
            printTable("GPT-2 XL Inference", gpt2EvalCpu, gpt2EvalGpu, gpt2EvalSmart)
            
            // B. Training
            gpt2.train()
            IntelligentRouter.shared.forcedBackend = .cpu
            IntelligentRouter.shared.enableSplitting = false
            let gpt2TrainCpu = try runTrainStep(gpt2, gpt2Opt, gpt2Input, gpt2Y)
            
            IntelligentRouter.shared.forcedBackend = .gpu
            IntelligentRouter.shared.enableSplitting = false
            let gpt2TrainGpu = try runTrainStep(gpt2, gpt2Opt, gpt2Input, gpt2Y)
            
            IntelligentRouter.shared.forcedBackend = nil
            IntelligentRouter.shared.enableSplitting = true
            let gpt2TrainSmart = try runTrainStep(gpt2, gpt2Opt, gpt2Input, gpt2Y)
            printTable("GPT-2 XL Training Step", gpt2TrainCpu, gpt2TrainGpu, gpt2TrainSmart)
        }
        
        func runMLP() throws {
            print("\n📝 MODEL 3: Deep MLP Block (Wider Dense Classifier)")
            print("   Batch: 1024 | Features: 4096 -> 4096 -> 10")
            print("-" .padding(toLength: 65, withPad: "-", startingAt: 0))
            
            let mlp = try GradSequential(
                try Fusion.nn.linear(4096, 4096),
                Fusion.nn.relu(),
                try Fusion.nn.linear(4096, 10)
            )
            let mlpX = try Fusion.rand([1024, 4096])
            let mlpY = try Tensor(shape: [1024], dtype: .float32)
            let mlpInput = GradTensor(mlpX, requiresGrad: false)
            let mlpOpt = Fusion.optim.adam(mlp.parameters(), lr: 0.01)
            
            // A. Inference
            mlp.eval()
            IntelligentRouter.shared.forcedBackend = .cpu
            IntelligentRouter.shared.enableSplitting = false
            let mlpEvalCpu = try runEval(mlp, mlpInput)
            
            IntelligentRouter.shared.forcedBackend = .gpu
            IntelligentRouter.shared.enableSplitting = false
            let mlpEvalGpu = try runEval(mlp, mlpInput)
            
            IntelligentRouter.shared.forcedBackend = nil
            IntelligentRouter.shared.enableSplitting = true
            let mlpEvalSmart = try runEval(mlp, mlpInput)
            printTable("MLP Inference", mlpEvalCpu, mlpEvalGpu, mlpEvalSmart)
            
            // B. Training
            mlp.train()
            IntelligentRouter.shared.forcedBackend = .cpu
            IntelligentRouter.shared.enableSplitting = false
            let mlpTrainCpu = try runTrainStep(mlp, mlpOpt, mlpInput, mlpY)
            
            IntelligentRouter.shared.forcedBackend = .gpu
            IntelligentRouter.shared.enableSplitting = false
            let mlpTrainGpu = try runTrainStep(mlp, mlpOpt, mlpInput, mlpY)
            
            IntelligentRouter.shared.forcedBackend = nil
            IntelligentRouter.shared.enableSplitting = true
            let mlpTrainSmart = try runTrainStep(mlp, mlpOpt, mlpInput, mlpY)
            printTable("MLP Training Step", mlpTrainCpu, mlpTrainGpu, mlpTrainSmart)
        }
        
        try autoreleasepool {
            try runLlama3()
        }
        MemoryManager.shared.clearPool()
        
        try autoreleasepool {
            try runGPT2()
        }
        MemoryManager.shared.clearPool()
        
        try autoreleasepool {
            try runMLP()
        }
        MemoryManager.shared.clearPool()
        
        // C. Square Matmul 3-Way Parallel Split evaluations
        func runSquareMatmul(size: Int) throws {
            print("\n📐 Square Matrix Multiplication: \(size)×\(size)")
            print("-" .padding(toLength: 65, withPad: "-", startingAt: 0))
            
            let a = try Tensor.random([size, size])
            let b = try Tensor.random([size, size])
            
            // Warmup
            for _ in 0..<3 { _ = try Tensor.matmul(a, b) }
            
            // Measure CPU
            let startCpu = CFAbsoluteTimeGetCurrent()
            for _ in 0..<5 { _ = try Tensor.matmul(a, b) }
            let cpuTime = (CFAbsoluteTimeGetCurrent() - startCpu) * 1000 / 5
            
            // Measure GPU
            let startGpu = CFAbsoluteTimeGetCurrent()
            for _ in 0..<5 { _ = try GPUEngine.shared.matmulMPS(a, b) }
            let gpuTime = (CFAbsoluteTimeGetCurrent() - startGpu) * 1000 / 5
            
            // Measure 2-Way Smart Split
            IntelligentRouter.shared.useThreeWaySplit = false
            IntelligentRouter.shared.enableSplitting = true
            let startSmart2 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<5 { _ = try IntelligentRouter.shared.matmul(a, b) }
            let smart2Time = (CFAbsoluteTimeGetCurrent() - startSmart2) * 1000 / 5
            
            // Measure 3-Way Smart Split
            IntelligentRouter.shared.useThreeWaySplit = true
            IntelligentRouter.shared.enableSplitting = true
            let startSmart3 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<5 { _ = try IntelligentRouter.shared.matmul(a, b) }
            let smart3Time = (CFAbsoluteTimeGetCurrent() - startSmart3) * 1000 / 5
            
            let bestSingle = min(cpuTime, gpuTime)
            let speedup2 = ((bestSingle - smart2Time) / bestSingle) * 100
            let speedup3 = ((bestSingle - smart3Time) / bestSingle) * 100
            
            print("   CPU-Only:      \(String(format: "%6.2f", cpuTime)) ms")
            print("   GPU-Only:      \(String(format: "%6.2f", gpuTime)) ms")
            print("   2-Way Split:   \(String(format: "%6.2f", smart2Time)) ms (Speedup: \(String(format: "%.1f", speedup2))%)")
            print("   3-Way Split:   \(String(format: "%6.2f", smart3Time)) ms ⚡ (Speedup: \(String(format: "%.1f", speedup3))%)")
        }
        
        try autoreleasepool {
            try runSquareMatmul(size: 512)
            try runSquareMatmul(size: 1024)
            try runSquareMatmul(size: 2048)
        }
        MemoryManager.shared.clearPool()
        
        print("\n" + "=" .padding(toLength: 65, withPad: "-", startingAt: 0))
        print("✅ All Model Benchmarks Complete!")
    }
}
