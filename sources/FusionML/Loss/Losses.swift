// Loss Functions - Common loss functions with gradient support
// MSE, CrossEntropy, BCE, NLL

import Foundation
import Accelerate

// MARK: - Loss Protocol

public protocol Loss {
    func forward(_ predictions: GradTensor, _ targets: Tensor) throws -> GradTensor
}

// MARK: - Mean Squared Error

/// MSE Loss: L = mean((pred - target)^2)
public final class MSELoss: Loss {
    public enum Reduction {
        case mean
        case sum
        case none
    }
    
    public let reduction: Reduction
    
    public init(reduction: Reduction = .mean) {
        self.reduction = reduction
    }
    
    public func forward(_ predictions: GradTensor, _ targets: Tensor) throws -> GradTensor {
        // Compute (pred - target)^2
        let predData = predictions.data.toArray()
        let targetData = targets.toArray()
        
        guard predData.count == targetData.count else {
            throw AutogradError.shapeMismatch("Predictions and targets must have same shape")
        }
        
        var diff = try Tensor(shape: predictions.shape, dtype: predictions.data.dtype)
        let diffPtr = diff.buffer.pointer.bindMemory(to: Float.self, capacity: diff.count)
        
        for i in 0..<predData.count {
            let d = predData[i] - targetData[i]
            diffPtr[i] = d * d
        }
        
        let diffGrad = GradTensor(diff, requiresGrad: predictions.requiresGrad)
        diffGrad.isLeaf = false
        
        // Set up gradient
        if diffGrad.requiresGrad {
            let targetsCopy = targets
            diffGrad.gradNode = GradNode(
                inputs: [predictions],
                gradFn: { inputs, gradOutput in
                    let pred = inputs[0]
                    let predArr = pred.toArray()
                    let targArr = targetsCopy.toArray()
                    var grad = try! Tensor(shape: pred.shape, dtype: pred.dtype)
                    let gradPtr = grad.buffer.pointer.bindMemory(to: Float.self, capacity: grad.count)
                    let upstreamArr = gradOutput.toArray()
                    let n = Float(pred.count)
                    
                    for i in 0..<pred.count {
                        // d/dp((p-t)^2) = 2(p-t)
                        gradPtr[i] = upstreamArr[i] * 2 * (predArr[i] - targArr[i]) / n
                    }
                    
                    return [grad]
                },
                name: "mse"
            )
        }
        
        // Reduce
        switch reduction {
        case .mean:
            return try diffGrad.mean()
        case .sum:
            return try diffGrad.sum()
        case .none:
            return diffGrad
        }
    }
}

// MARK: - Cross Entropy Loss

/// CrossEntropy Loss for classification
public final class CrossEntropyLoss: Loss {
    public init() {}
    
    public func forward(_ predictions: GradTensor, _ targets: Tensor) throws -> GradTensor {
        // predictions: [batch, classes] (logits)
        // targets: [batch] (class indices as floats)
        
        let batch = predictions.shape[0]
        let classes = predictions.shape[1]
        
        // Softmax
        let logits = predictions.data.toArray()
        var probs = [Float](repeating: 0, count: logits.count)
        
        for b in 0..<batch {
            let offset = b * classes
            
            // Find max for numerical stability
            var maxVal: Float = -Float.infinity
            for c in 0..<classes {
                maxVal = max(maxVal, logits[offset + c])
            }
            
            // Compute exp and sum
            var sumExp: Float = 0
            for c in 0..<classes {
                probs[offset + c] = exp(logits[offset + c] - maxVal)
                sumExp += probs[offset + c]
            }
            
            // Normalize
            for c in 0..<classes {
                probs[offset + c] /= sumExp
            }
        }
        
        // Compute loss: -log(p[target])
        let targetData = targets.toArray()
        var loss: Float = 0
        
        for b in 0..<batch {
            let targetClass = Int(targetData[b])
            let prob = probs[b * classes + targetClass]
            loss -= log(max(prob, 1e-7))
        }
        loss /= Float(batch)
        
        let result = try Tensor(shape: [1], dtype: .float32)
        result.buffer.pointer.bindMemory(to: Float.self, capacity: 1).pointee = loss
        
        let output = GradTensor(result, requiresGrad: predictions.requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            let probsCopy = probs
            let targetsCopy = targets
            let batchSize = batch
            let numClasses = classes
            
            output.gradNode = GradNode(
                inputs: [predictions],
                gradFn: { _, gradOutput in
                    let targArr = targetsCopy.toArray()
                    let upstream = gradOutput.toArray()[0]
                    
                    var grad = try! Tensor(shape: [batchSize, numClasses], dtype: .float32)
                    let gradPtr = grad.buffer.pointer.bindMemory(to: Float.self, capacity: grad.count)
                    
                    for b in 0..<batchSize {
                        let targetClass = Int(targArr[b])
                        for c in 0..<numClasses {
                            let p = probsCopy[b * numClasses + c]
                            let indicator: Float = (c == targetClass) ? 1.0 : 0.0
                            gradPtr[b * numClasses + c] = upstream * (p - indicator) / Float(batchSize)
                        }
                    }
                    
                    return [grad]
                },
                name: "cross_entropy"
            )
        }
        
        return output
    }
}

// MARK: - Binary Cross Entropy

/// BCE Loss for binary classification
public final class BCELoss: Loss {
    public init() {}
    
    public func forward(_ predictions: GradTensor, _ targets: Tensor) throws -> GradTensor {
        // predictions: probabilities [0, 1]
        // targets: binary labels
        
        let predData = predictions.data.toArray()
        let targetData = targets.toArray()
        
        var loss: Float = 0
        for i in 0..<predData.count {
            let p = max(min(predData[i], 1 - 1e-7), 1e-7)
            let t = targetData[i]
            loss -= t * log(p) + (1 - t) * log(1 - p)
        }
        loss /= Float(predData.count)
        
        let result = try Tensor(shape: [1], dtype: .float32)
        result.buffer.pointer.bindMemory(to: Float.self, capacity: 1).pointee = loss
        
        let output = GradTensor(result, requiresGrad: predictions.requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            let targetsCopy = targets
            output.gradNode = GradNode(
                inputs: [predictions],
                gradFn: { inputs, gradOutput in
                    let pred = inputs[0]
                    let predArr = pred.toArray()
                    let targArr = targetsCopy.toArray()
                    let upstream = gradOutput.toArray()[0]
                    let n = Float(pred.count)
                    
                    var grad = try! Tensor(shape: pred.shape, dtype: pred.dtype)
                    let gradPtr = grad.buffer.pointer.bindMemory(to: Float.self, capacity: grad.count)
                    
                    for i in 0..<pred.count {
                        let p = max(min(predArr[i], 1 - 1e-7), 1e-7)
                        let t = targArr[i]
                        // d/dp(-t*log(p) - (1-t)*log(1-p)) = -t/p + (1-t)/(1-p)
                        gradPtr[i] = upstream * (-t / p + (1 - t) / (1 - p)) / n
                    }
                    
                    return [grad]
                },
                name: "bce"
            )
        }
        
        return output
    }
}

// MARK: - L1 Loss

/// L1 (Mean Absolute Error) Loss
public final class L1Loss: Loss {
    public init() {}
    
    public func forward(_ predictions: GradTensor, _ targets: Tensor) throws -> GradTensor {
        let predData = predictions.data.toArray()
        let targetData = targets.toArray()
        
        var loss: Float = 0
        for i in 0..<predData.count {
            loss += abs(predData[i] - targetData[i])
        }
        loss /= Float(predData.count)
        
        let result = try Tensor(shape: [1], dtype: .float32)
        result.buffer.pointer.bindMemory(to: Float.self, capacity: 1).pointee = loss
        
        let output = GradTensor(result, requiresGrad: predictions.requiresGrad)
        output.isLeaf = false
        
        if output.requiresGrad {
            let targetsCopy = targets
            output.gradNode = GradNode(
                inputs: [predictions],
                gradFn: { inputs, gradOutput in
                    let pred = inputs[0]
                    let predArr = pred.toArray()
                    let targArr = targetsCopy.toArray()
                    let upstream = gradOutput.toArray()[0]
                    let n = Float(pred.count)
                    
                    var grad = try! Tensor(shape: pred.shape, dtype: pred.dtype)
                    let gradPtr = grad.buffer.pointer.bindMemory(to: Float.self, capacity: grad.count)
                    
                    for i in 0..<pred.count {
                        let diff = predArr[i] - targArr[i]
                        gradPtr[i] = upstream * (diff > 0 ? 1 : -1) / n
                    }
                    
                    return [grad]
                },
                name: "l1"
            )
        }
        
        return output
    }
}
