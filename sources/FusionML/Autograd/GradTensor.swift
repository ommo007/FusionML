// GradTensor - Tensor with Automatic Differentiation
// Tracks gradients and builds computation graph for backpropagation

import Foundation
import Accelerate

/// Gradient function type - computes gradients for backward pass
public typealias GradFn = ([Tensor], Tensor) -> [Tensor]

/// Node in the computation graph
public final class GradNode: @unchecked Sendable {
    public let inputs: [GradTensor]
    public let gradFn: GradFn?
    public let name: String
    
    public init(inputs: [GradTensor], gradFn: GradFn?, name: String = "op") {
        self.inputs = inputs
        self.gradFn = gradFn
        self.name = name
    }
}

/// Tensor with gradient tracking for automatic differentiation
public final class GradTensor: @unchecked Sendable {
    
    /// The underlying data tensor
    public var data: Tensor
    
    /// Gradient tensor (populated after backward())
    public var grad: Tensor?
    
    /// Whether this tensor requires gradient computation
    public var requiresGrad: Bool
    
    /// Computation graph node (for backward pass)
    public var gradNode: GradNode?
    
    /// Whether this is a leaf tensor (user-created, not from operation)
    public var isLeaf: Bool
    
    // MARK: - Initialization
    
    public init(_ data: Tensor, requiresGrad: Bool = false) {
        self.data = data
        self.requiresGrad = requiresGrad
        self.grad = nil
        self.gradNode = nil
        self.isLeaf = true
    }
    
    public init(shape: [Int], requiresGrad: Bool = false) throws {
        self.data = try Tensor.zeros(shape)
        self.requiresGrad = requiresGrad
        self.grad = nil
        self.gradNode = nil
        self.isLeaf = true
    }
    
    // MARK: - Factory Methods
    
    public static func zeros(_ shape: [Int], requiresGrad: Bool = false) throws -> GradTensor {
        GradTensor(try Tensor.zeros(shape), requiresGrad: requiresGrad)
    }
    
    public static func ones(_ shape: [Int], requiresGrad: Bool = false) throws -> GradTensor {
        GradTensor(try Tensor.ones(shape), requiresGrad: requiresGrad)
    }
    
    public static func random(_ shape: [Int], requiresGrad: Bool = false) throws -> GradTensor {
        GradTensor(try Tensor.random(shape), requiresGrad: requiresGrad)
    }
    
    /// Create from Tensor, setting up for gradient computation
    public static func from(_ tensor: Tensor, requiresGrad: Bool = true) -> GradTensor {
        GradTensor(tensor, requiresGrad: requiresGrad)
    }
    
    // MARK: - Backward Pass
    
    /// Compute gradients via backpropagation
    public func backward(_ gradOutput: Tensor? = nil) throws {
        guard requiresGrad else {
            throw AutogradError.noGradient("Tensor does not require gradients")
        }
        
        // Default gradient is ones (for scalar loss)
        let upstream: Tensor
        if let grad = gradOutput {
            upstream = grad
        } else {
            upstream = try Tensor.ones(data.shape)
        }
        
        // Topological sort of nodes
        var sorted: [GradTensor] = []
        var visited = Set<ObjectIdentifier>()
        
        func visit(_ t: GradTensor) {
            let id = ObjectIdentifier(t)
            if visited.contains(id) { return }
            visited.insert(id)
            
            if let node = t.gradNode {
                for input in node.inputs {
                    visit(input)
                }
            }
            sorted.append(t)
        }
        
        visit(self)
        sorted.reverse()  // Process from output to inputs
        
        // Initialize this tensor's gradient
        self.grad = upstream
        
        // Backpropagate through the graph
        for tensor in sorted {
            guard let node = tensor.gradNode,
                  let gradFn = node.gradFn,
                  let grad = tensor.grad else { continue }
            
            let inputData = node.inputs.map { $0.data }
            let inputGrads = gradFn(inputData, grad)
            
            // Accumulate gradients to inputs
            for (input, inputGrad) in zip(node.inputs, inputGrads) {
                if input.requiresGrad {
                    if input.grad == nil {
                        input.grad = inputGrad
                    } else {
                        // Accumulate gradients
                        input.grad = try Tensor.add(input.grad!, inputGrad)
                    }
                }
            }
        }
    }
    
    /// Zero out gradients
    public func zeroGrad() {
        grad = nil
    }
    
    /// Detach from computation graph
    public func detach() -> GradTensor {
        let detached = GradTensor(data, requiresGrad: false)
        detached.isLeaf = true
        detached.gradNode = nil
        return detached
    }
    
    // MARK: - Properties
    
    public var shape: [Int] { data.shape }
    public var count: Int { data.count }
    public var ndim: Int { data.ndim }
}

// MARK: - Errors

public enum AutogradError: Error {
    case noGradient(String)
    case shapeMismatch(String)
    case notImplemented(String)
}
