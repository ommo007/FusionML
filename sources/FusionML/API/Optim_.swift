// Fusion.optim - Optimizer Module
// PyTorch-style optimizer API

import Foundation

extension Fusion {
    
    /// Optimizer module
    public enum optim {
        
        // MARK: - Factory Methods
        
        /// Create SGD optimizer
        public static func sgd(
            _ parameters: [GradTensor],
            lr: Float = 0.01,
            momentum: Float = 0.0,
            weightDecay: Float = 0.0,
            nesterov: Bool = false
        ) -> SGD {
            SGD(
                parameters: parameters,
                lr: lr,
                momentum: momentum,
                weightDecay: weightDecay,
                nesterov: nesterov
            )
        }
        
        /// Create Adam optimizer
        public static func adam(
            _ parameters: [GradTensor],
            lr: Float = 0.001,
            beta1: Float = 0.9,
            beta2: Float = 0.999,
            eps: Float = 1e-8,
            weightDecay: Float = 0.0
        ) -> Adam {
            Adam(
                parameters: parameters,
                lr: lr,
                beta1: beta1,
                beta2: beta2,
                epsilon: eps,
                weightDecay: weightDecay
            )
        }
        
        /// Create AdamW optimizer (Adam with decoupled weight decay)
        public static func adamw(
            _ parameters: [GradTensor],
            lr: Float = 0.001,
            beta1: Float = 0.9,
            beta2: Float = 0.999,
            eps: Float = 1e-8,
            weightDecay: Float = 0.01
        ) -> Adam {
            Adam(
                parameters: parameters,
                lr: lr,
                beta1: beta1,
                beta2: beta2,
                epsilon: eps,
                weightDecay: weightDecay
            )
        }
    }
}

// MARK: - Fusion.optim.lr_scheduler

extension Fusion.optim {
    
    /// Learning rate schedulers
    public enum lr_scheduler {
        
        /// Create step scheduler
        public static func step(_ optimizer: Optimizer, stepSize: Int, gamma: Float = 0.1) -> StepLR {
            StepLR(optimizer: optimizer, stepSize: stepSize, gamma: gamma)
        }
        
        /// Create cosine annealing scheduler
        public static func cosine(_ optimizer: Optimizer, tMax: Int, etaMin: Float = 0) -> CosineAnnealingLR {
            CosineAnnealingLR(optimizer: optimizer, tMax: tMax, etaMin: etaMin)
        }
    }
}

