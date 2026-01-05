// Fusion.data - Data Loading Module
// Dataset and DataLoader utilities

import Foundation

extension Fusion {
    
    /// Data loading module
    public enum data {
        
        // MARK: - Types
        
        // Types are directly accessible without prefix
        // Dataset, TensorDataset, DataLoader are all in global scope
        

        
        // MARK: - Factory Methods
        
        /// Create tensor dataset
        public static func tensorDataset(data: Tensor, labels: Tensor) -> TensorDataset {
            TensorDataset(data: data, labels: labels)
        }
        
        /// Create data loader
        public static func dataLoader<D: Dataset>(
            _ dataset: D,
            batchSize: Int,
            shuffle: Bool = false,
            dropLast: Bool = false
        ) -> DataLoader<D> {
            DataLoader(
                dataset: dataset,
                batchSize: batchSize,
                shuffle: shuffle,
                dropLast: dropLast
            )
        }
        
        /// Collate tensor batches
        public static func collate(_ batch: [(Tensor, Tensor)]) throws -> (Tensor, Tensor) {
            try collateTensors(batch)
        }
    }
}

// MARK: - Fusion.random

extension Fusion {
    
    /// Random number generation
    public enum random {
        
        /// Set random seed
        public static func seed(_ value: UInt64) {
            srand48(Int(value))
        }
        
        /// Create tensor with uniform random values [0, 1)
        public static func rand(_ shape: [Int]) throws -> Tensor {
            let tensor = try Tensor(shape: shape, dtype: .float32)
            let ptr = tensor.buffer.pointer.bindMemory(to: Float.self, capacity: tensor.count)
            for i in 0..<tensor.count {
                ptr[i] = Float.random(in: 0..<1)
            }
            return tensor
        }
        
        /// Create tensor with normal random values (mean=0, std=1)
        public static func randn(_ shape: [Int]) throws -> Tensor {
            let tensor = try Tensor(shape: shape, dtype: .float32)
            let ptr = tensor.buffer.pointer.bindMemory(to: Float.self, capacity: tensor.count)
            
            for i in 0..<tensor.count {
                // Box-Muller transform
                let u1 = Float.random(in: 0.0001...1)
                let u2 = Float.random(in: 0...1)
                ptr[i] = sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
            }
            return tensor
        }
        
        /// Create tensor with uniform random integers
        public static func randint(_ low: Int, _ high: Int, _ shape: [Int]) throws -> Tensor {
            let tensor = try Tensor(shape: shape, dtype: .float32)
            let ptr = tensor.buffer.pointer.bindMemory(to: Float.self, capacity: tensor.count)
            for i in 0..<tensor.count {
                ptr[i] = Float(Int.random(in: low..<high))
            }
            return tensor
        }
    }
}
