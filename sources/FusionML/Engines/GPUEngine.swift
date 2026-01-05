// SiliconML - GPU Engine
// High-performance GPU compute using custom Metal shaders

import Metal
import MetalPerformanceShaders

/// GPU compute engine with custom shaders
public final class GPUEngine: @unchecked Sendable {
    
    public static let shared = GPUEngine()
    
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private var library: MTLLibrary?
    private var pipelines: [String: MTLComputePipelineState] = [:]
    
    private init() {
        self.device = MemoryManager.shared.device
        self.commandQueue = MemoryManager.shared.commandQueue
        loadShaders()
    }
    
    private func loadShaders() {
        // Try to load from bundle or compile from source
        do {
            // First try default library
            if let lib = device.makeDefaultLibrary() {
                self.library = lib
            } else {
                // Compile shaders from source at runtime
                let shaderSource = GPUEngine.shaderSource
                self.library = try device.makeLibrary(source: shaderSource, options: nil)
            }
            
            // Pre-compile pipeline states
            try compilePipeline(name: "matmul_tiled")
            try compilePipeline(name: "matmul_naive")
            try compilePipeline(name: "matmul_fp16_tiled")
            try compilePipeline(name: "add_elementwise")
            try compilePipeline(name: "mul_elementwise")
            try compilePipeline(name: "relu")
            try compilePipeline(name: "gelu")
            try compilePipeline(name: "softmax_row")
            
        } catch {
            print("Warning: Failed to load shaders: \(error)")
        }
    }
    
    private func compilePipeline(name: String) throws {
        guard let library = library,
              let function = library.makeFunction(name: name) else {
            return
        }
        pipelines[name] = try device.makeComputePipelineState(function: function)
    }
    
    // MARK: - Matrix Operations
    
    /// Matrix multiplication using custom tiled shader
    public func matmul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        guard a.ndim == 2 && b.ndim == 2 else {
            throw MemoryError.invalidShape
        }
        guard a.shape[1] == b.shape[0] else {
            throw MemoryError.invalidShape
        }
        
        let M = a.shape[0]
        let K = a.shape[1]
        let N = b.shape[1]
        
        let result = try Tensor(shape: [M, N], dtype: a.dtype)
        
        guard let pipeline = pipelines["matmul_tiled"],
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MemoryError.deviceNotAvailable
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(a.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(b.metalBuffer, offset: 0, index: 1)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 2)
        
        var mVal = Int32(M)
        var kVal = Int32(K)
        var nVal = Int32(N)
        encoder.setBytes(&mVal, length: 4, index: 3)
        encoder.setBytes(&kVal, length: 4, index: 4)
        encoder.setBytes(&nVal, length: 4, index: 5)
        
        let tileSize = 32
        let threadgroupSize = MTLSize(width: tileSize, height: tileSize, depth: 1)
        let gridSize = MTLSize(
            width: (N + tileSize - 1) / tileSize * tileSize,
            height: (M + tileSize - 1) / tileSize * tileSize,
            depth: 1
        )
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return result
    }
    
    /// Matrix multiplication using MPS (for comparison)
    public func matmulMPS(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        guard a.ndim == 2 && b.ndim == 2 else {
            throw MemoryError.invalidShape
        }
        guard a.shape[1] == b.shape[0] else {
            throw MemoryError.invalidShape
        }
        
        let M = a.shape[0]
        let K = a.shape[1]
        let N = b.shape[1]
        
        let result = try Tensor(shape: [M, N], dtype: .float32)
        
        let aDesc = MPSMatrixDescriptor(rows: M, columns: K, rowBytes: K * 4, dataType: .float32)
        let bDesc = MPSMatrixDescriptor(rows: K, columns: N, rowBytes: N * 4, dataType: .float32)
        let cDesc = MPSMatrixDescriptor(rows: M, columns: N, rowBytes: N * 4, dataType: .float32)
        
        let aMatrix = MPSMatrix(buffer: a.metalBuffer, descriptor: aDesc)
        let bMatrix = MPSMatrix(buffer: b.metalBuffer, descriptor: bDesc)
        let cMatrix = MPSMatrix(buffer: result.metalBuffer, descriptor: cDesc)
        
        let matmul = MPSMatrixMultiplication(device: device, resultRows: M, resultColumns: N, interiorColumns: K)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MemoryError.deviceNotAvailable
        }
        
        matmul.encode(commandBuffer: commandBuffer, leftMatrix: aMatrix, rightMatrix: bMatrix, resultMatrix: cMatrix)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return result
    }
    
    // MARK: - Element-wise Operations
    
    private func elementwise(_ a: Tensor, _ b: Tensor, kernel: String) throws -> Tensor {
        guard a.shape == b.shape else {
            throw MemoryError.invalidShape
        }
        
        let result = try Tensor(shape: a.shape, dtype: a.dtype)
        
        guard let pipeline = pipelines[kernel],
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MemoryError.deviceNotAvailable
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(a.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(b.metalBuffer, offset: 0, index: 1)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 2)
        
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (a.count + threadsPerGroup - 1) / threadsPerGroup
        
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return result
    }
    
    public func add(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        try elementwise(a, b, kernel: "add_elementwise")
    }
    
    public func mul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        try elementwise(a, b, kernel: "mul_elementwise")
    }
    
    // MARK: - Activation Functions
    
    public func relu(_ x: Tensor) throws -> Tensor {
        let result = try Tensor(shape: x.shape, dtype: x.dtype)
        
        guard let pipeline = pipelines["relu"],
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MemoryError.deviceNotAvailable
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(x.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 1)
        
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (x.count + threadsPerGroup - 1) / threadsPerGroup
        
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return result
    }
    
    public func gelu(_ x: Tensor) throws -> Tensor {
        let result = try Tensor(shape: x.shape, dtype: x.dtype)
        
        guard let pipeline = pipelines["gelu"],
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MemoryError.deviceNotAvailable
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(x.metalBuffer, offset: 0, index: 0)
        encoder.setBuffer(result.metalBuffer, offset: 0, index: 1)
        
        let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (x.count + threadsPerGroup - 1) / threadsPerGroup
        
        encoder.dispatchThreadgroups(MTLSize(width: threadgroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return result
    }
    
    // MARK: - Embedded Shader Source
    
    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;
    
    constant int TILE_SIZE = 32;
    
    kernel void matmul_naive(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant int& M [[buffer(3)]],
        constant int& K [[buffer(4)]],
        constant int& N [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        int row = gid.y;
        int col = gid.x;
        if (row >= M || col >= N) return;
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
    
    kernel void matmul_tiled(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant int& M [[buffer(3)]],
        constant int& K [[buffer(4)]],
        constant int& N [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]],
        uint2 tid [[thread_position_in_threadgroup]],
        uint2 tgid [[threadgroup_position_in_grid]]
    ) {
        threadgroup float As[TILE_SIZE][TILE_SIZE];
        threadgroup float Bs[TILE_SIZE][TILE_SIZE];
        
        int row = tgid.y * TILE_SIZE + tid.y;
        int col = tgid.x * TILE_SIZE + tid.x;
        float sum = 0.0f;
        
        int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
        for (int t = 0; t < numTiles; t++) {
            int aCol = t * TILE_SIZE + tid.x;
            As[tid.y][tid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
            
            int bRow = t * TILE_SIZE + tid.y;
            Bs[tid.y][tid.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[tid.y][k] * Bs[k][tid.x];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        if (row < M && col < N) {
            C[row * N + col] = sum;
        }
    }
    
    kernel void matmul_fp16_tiled(
        device const half* A [[buffer(0)]],
        device const half* B [[buffer(1)]],
        device half* C [[buffer(2)]],
        constant int& M [[buffer(3)]],
        constant int& K [[buffer(4)]],
        constant int& N [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]],
        uint2 tid [[thread_position_in_threadgroup]],
        uint2 tgid [[threadgroup_position_in_grid]]
    ) {
        threadgroup half As[TILE_SIZE][TILE_SIZE];
        threadgroup half Bs[TILE_SIZE][TILE_SIZE];
        
        int row = tgid.y * TILE_SIZE + tid.y;
        int col = tgid.x * TILE_SIZE + tid.x;
        half sum = 0.0h;
        
        int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
        for (int t = 0; t < numTiles; t++) {
            int aCol = t * TILE_SIZE + tid.x;
            As[tid.y][tid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0h;
            
            int bRow = t * TILE_SIZE + tid.y;
            Bs[tid.y][tid.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0h;
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[tid.y][k] * Bs[k][tid.x];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        if (row < M && col < N) {
            C[row * N + col] = sum;
        }
    }
    
    kernel void add_elementwise(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        C[id] = A[id] + B[id];
    }
    
    kernel void mul_elementwise(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        C[id] = A[id] * B[id];
    }
    
    kernel void relu(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        output[id] = max(input[id], 0.0f);
    }
    
    kernel void gelu(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        uint id [[thread_position_in_grid]]
    ) {
        float x = input[id];
        float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[id] = x * cdf;
    }
    
    kernel void softmax_row(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant int& cols [[buffer(2)]],
        uint row [[thread_position_in_grid]]
    ) {
        int offset = row * cols;
        float maxVal = input[offset];
        for (int i = 1; i < cols; i++) {
            maxVal = max(maxVal, input[offset + i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            float val = exp(input[offset + i] - maxVal);
            output[offset + i] = val;
            sum += val;
        }
        for (int i = 0; i < cols; i++) {
            output[offset + i] /= sum;
        }
    }
    """
}
