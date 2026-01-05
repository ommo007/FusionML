// BackwardOps - Backward implementations for automatic differentiation
// Each operation has a corresponding gradient function

import Foundation
import Accelerate

// MARK: - Differentiable Operations on GradTensor

extension GradTensor {
    
    // MARK: - Matrix Multiplication
    
    /// Matrix multiply with gradient tracking: C = A @ B
    public static func matmul(_ a: GradTensor, _ b: GradTensor) throws -> GradTensor {
        // Forward pass - use intelligent router
        let result = try IntelligentRouter.shared.matmul(a.data, b.data)
        let output = GradTensor(result, requiresGrad: a.requiresGrad || b.requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            output.gradNode = GradNode(
                inputs: [a, b],
                gradFn: { inputs, gradOutput in
                    // dL/dA = gradOutput @ B^T
                    // dL/dB = A^T @ gradOutput
                    let aData = inputs[0]
                    let bData = inputs[1]
                    
                    var gradA: Tensor
                    var gradB: Tensor
                    
                    do {
                        // Transpose B for grad_A
                        let bT = try Tensor.transpose(bData)
                        gradA = try IntelligentRouter.shared.matmul(gradOutput, bT)
                        
                        // Transpose A for grad_B
                        let aT = try Tensor.transpose(aData)
                        gradB = try IntelligentRouter.shared.matmul(aT, gradOutput)
                    } catch {
                        gradA = gradOutput  // Fallback
                        gradB = gradOutput
                    }
                    
                    return [gradA, gradB]
                },
                name: "matmul"
            )
        }
        
        return output
    }
    
    // MARK: - Element-wise Addition
    
    /// Element-wise add with gradient tracking
    public static func add(_ a: GradTensor, _ b: GradTensor) throws -> GradTensor {
        let result = try IntelligentRouter.shared.add(a.data, b.data)
        let output = GradTensor(result, requiresGrad: a.requiresGrad || b.requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            output.gradNode = GradNode(
                inputs: [a, b],
                gradFn: { _, gradOutput in
                    // Gradient flows through unchanged for addition
                    return [gradOutput, gradOutput]
                },
                name: "add"
            )
        }
        
        return output
    }
    
    // MARK: - Element-wise Multiplication
    
    /// Element-wise multiply with gradient tracking
    public static func mul(_ a: GradTensor, _ b: GradTensor) throws -> GradTensor {
        let result = try Tensor.mul(a.data, b.data)
        let output = GradTensor(result, requiresGrad: a.requiresGrad || b.requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            output.gradNode = GradNode(
                inputs: [a, b],
                gradFn: { inputs, gradOutput in
                    // dL/dA = gradOutput * B
                    // dL/dB = gradOutput * A
                    do {
                        let gradA = try Tensor.mul(gradOutput, inputs[1])
                        let gradB = try Tensor.mul(gradOutput, inputs[0])
                        return [gradA, gradB]
                    } catch {
                        return [gradOutput, gradOutput]
                    }
                },
                name: "mul"
            )
        }
        
        return output
    }
    
    // MARK: - ReLU Activation
    
    /// ReLU with gradient tracking
    public func relu() throws -> GradTensor {
        let result = try IntelligentRouter.shared.relu(data)
        let output = GradTensor(result, requiresGrad: requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            output.gradNode = GradNode(
                inputs: [self],
                gradFn: { inputs, gradOutput in
                    // dL/dx = gradOutput * (x > 0 ? 1 : 0)
                    let input = inputs[0]
                    var mask = try! Tensor(shape: input.shape, dtype: input.dtype)
                    let inputData = input.toArray()
                    let maskPtr = mask.buffer.pointer.bindMemory(to: Float.self, capacity: mask.count)
                    
                    for i in 0..<input.count {
                        maskPtr[i] = inputData[i] > 0 ? 1.0 : 0.0
                    }
                    
                    return [try! Tensor.mul(gradOutput, mask)]
                },
                name: "relu"
            )
        }
        
        return output
    }
    
    // MARK: - GELU Activation
    
    /// GELU with gradient tracking
    public func gelu() throws -> GradTensor {
        let result = try IntelligentRouter.shared.gelu(data)
        let output = GradTensor(result, requiresGrad: requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            output.gradNode = GradNode(
                inputs: [self],
                gradFn: { inputs, gradOutput in
                    // GELU gradient is complex, using approximation
                    let x = inputs[0]
                    let xData = x.toArray()
                    var gradInput = try! Tensor(shape: x.shape, dtype: x.dtype)
                    let gradPtr = gradInput.buffer.pointer.bindMemory(to: Float.self, capacity: gradInput.count)
                    let upstreamData = gradOutput.toArray()
                    
                    for i in 0..<x.count {
                        let xi = xData[i]
                        // Approximate GELU derivative
                        let cdf = 0.5 * (1 + tanh(0.7978845608 * (xi + 0.044715 * xi * xi * xi)))
                        let pdf = exp(-0.5 * xi * xi) / sqrt(2 * .pi)
                        let grad = Float(cdf + xi * pdf)
                        gradPtr[i] = upstreamData[i] * grad
                    }
                    
                    return [gradInput]
                },
                name: "gelu"
            )
        }
        
        return output
    }
    
    // MARK: - Sum Reduction
    
    /// Sum all elements with gradient tracking
    public func sum() throws -> GradTensor {
        let data = self.data.toArray()
        let sumValue = data.reduce(0, +)
        let result = try Tensor(shape: [1], dtype: self.data.dtype)
        result.buffer.pointer.bindMemory(to: Float.self, capacity: 1).pointee = sumValue
        
        let output = GradTensor(result, requiresGrad: requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            let inputShape = self.shape
            output.gradNode = GradNode(
                inputs: [self],
                gradFn: { _, gradOutput in
                    // Gradient of sum is ones * upstream
                    var grad = try! Tensor.ones(inputShape)
                    let scale = gradOutput.toArray()[0]
                    let ptr = grad.buffer.pointer.bindMemory(to: Float.self, capacity: grad.count)
                    for i in 0..<grad.count {
                        ptr[i] = scale
                    }
                    return [grad]
                },
                name: "sum"
            )
        }
        
        return output
    }
    
    // MARK: - Mean Reduction
    
    /// Mean of all elements with gradient tracking
    public func mean() throws -> GradTensor {
        let dataArr = self.data.toArray()
        let meanValue = dataArr.reduce(0, +) / Float(dataArr.count)
        let result = try Tensor(shape: [1], dtype: self.data.dtype)
        result.buffer.pointer.bindMemory(to: Float.self, capacity: 1).pointee = meanValue
        
        let output = GradTensor(result, requiresGrad: requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            let inputShape = self.shape
            let inputCount = self.count
            output.gradNode = GradNode(
                inputs: [self],
                gradFn: { _, gradOutput in
                    // Gradient of mean is (1/n) * ones * upstream
                    var grad = try! Tensor(shape: inputShape, dtype: .float32)
                    let scale = gradOutput.toArray()[0] / Float(inputCount)
                    let ptr = grad.buffer.pointer.bindMemory(to: Float.self, capacity: grad.count)
                    for i in 0..<grad.count {
                        ptr[i] = scale
                    }
                    return [grad]
                },
                name: "mean"
            )
        }
        
        return output
    }
    
    // MARK: - Power
    
    /// Element-wise power with gradient tracking
    public func pow(_ exponent: Float) throws -> GradTensor {
        var result = try Tensor(shape: shape, dtype: data.dtype)
        let inputData = data.toArray()
        let resultPtr = result.buffer.pointer.bindMemory(to: Float.self, capacity: count)
        
        for i in 0..<count {
            resultPtr[i] = Foundation.pow(inputData[i], exponent)
        }
        
        let output = GradTensor(result, requiresGrad: requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            let exp = exponent
            output.gradNode = GradNode(
                inputs: [self],
                gradFn: { inputs, gradOutput in
                    // d/dx(x^n) = n * x^(n-1)
                    let x = inputs[0]
                    let xData = x.toArray()
                    var grad = try! Tensor(shape: x.shape, dtype: x.dtype)
                    let gradPtr = grad.buffer.pointer.bindMemory(to: Float.self, capacity: grad.count)
                    let upstreamData = gradOutput.toArray()
                    
                    for i in 0..<x.count {
                        gradPtr[i] = upstreamData[i] * exp * Foundation.pow(xData[i], exp - 1)
                    }
                    
                    return [grad]
                },
                name: "pow"
            )
        }
        
        return output
    }
}
