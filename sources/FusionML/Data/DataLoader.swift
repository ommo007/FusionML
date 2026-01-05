// Dataset and DataLoader - Data pipeline for training
// Batching, shuffling, parallel loading

import Foundation

// MARK: - Dataset Protocol

/// Protocol for datasets
public protocol Dataset {
    associatedtype Sample
    
    var count: Int { get }
    subscript(index: Int) -> Sample { get }
}

// MARK: - Tensor Dataset

/// Simple tensor dataset
public final class TensorDataset: Dataset {
    public typealias Sample = (Tensor, Tensor)
    
    public let data: Tensor
    public let labels: Tensor
    
    public var count: Int { data.shape[0] }
    
    public init(data: Tensor, labels: Tensor) {
        self.data = data
        self.labels = labels
    }
    
    public subscript(index: Int) -> (Tensor, Tensor) {
        // Extract single sample
        let numFeatures = data.count / data.shape[0]
        let dataOffset = index * numFeatures
        
        var sample = try! Tensor(shape: Array(data.shape.dropFirst()), dtype: data.dtype)
        memcpy(sample.buffer.pointer,
               data.buffer.pointer.advanced(by: dataOffset * 4),
               numFeatures * 4)
        
        // Extract label
        let labelSize = labels.count / labels.shape[0]
        let labelOffset = index * labelSize
        
        var label = try! Tensor(shape: Array(labels.shape.dropFirst()), dtype: labels.dtype)
        if labelSize == 1 {
            label = try! Tensor(shape: [1], dtype: labels.dtype)
        }
        memcpy(label.buffer.pointer,
               labels.buffer.pointer.advanced(by: labelOffset * 4),
               labelSize * 4)
        
        return (sample, label)
    }
}

// MARK: - DataLoader

/// DataLoader for batched iteration
public final class DataLoader<D: Dataset>: Sequence {
    public typealias Element = ([D.Sample])
    
    public let dataset: D
    public let batchSize: Int
    public let shuffle: Bool
    public let dropLast: Bool
    
    private var indices: [Int]
    
    public init(
        dataset: D,
        batchSize: Int,
        shuffle: Bool = false,
        dropLast: Bool = false
    ) {
        self.dataset = dataset
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.dropLast = dropLast
        self.indices = Array(0..<dataset.count)
    }
    
    public func makeIterator() -> DataLoaderIterator<D> {
        var indices = self.indices
        if shuffle {
            indices.shuffle()
        }
        return DataLoaderIterator(
            dataset: dataset,
            indices: indices,
            batchSize: batchSize,
            dropLast: dropLast
        )
    }
    
    public var count: Int {
        if dropLast {
            return dataset.count / batchSize
        }
        return (dataset.count + batchSize - 1) / batchSize
    }
}

/// Iterator for DataLoader
public struct DataLoaderIterator<D: Dataset>: IteratorProtocol {
    public typealias Element = [D.Sample]
    
    let dataset: D
    let indices: [Int]
    let batchSize: Int
    let dropLast: Bool
    var currentIndex: Int = 0
    
    public mutating func next() -> [D.Sample]? {
        guard currentIndex < indices.count else { return nil }
        
        let remainingCount = indices.count - currentIndex
        if dropLast && remainingCount < batchSize {
            return nil
        }
        
        let batchEnd = min(currentIndex + batchSize, indices.count)
        var batch: [D.Sample] = []
        
        for i in currentIndex..<batchEnd {
            batch.append(dataset[indices[i]])
        }
        
        currentIndex = batchEnd
        return batch
    }
}

// MARK: - Collate Functions

/// Collate tensor batches into single tensors
public func collateTensors(_ batch: [(Tensor, Tensor)]) throws -> (Tensor, Tensor) {
    guard !batch.isEmpty else {
        throw DataError.emptyBatch
    }
    
    let batchSize = batch.count
    let sampleShape = batch[0].0.shape
    let labelShape = batch[0].1.shape
    
    // Create batched tensors
    let dataShape = [batchSize] + sampleShape
    let labelShape2 = [batchSize] + labelShape
    
    let batchedData = try Tensor(shape: dataShape, dtype: batch[0].0.dtype)
    let batchedLabels = try Tensor(shape: labelShape2, dtype: batch[0].1.dtype)
    
    let sampleSize = sampleShape.reduce(1, *)
    let labelSize = labelShape.reduce(1, *)
    
    for (i, (sample, label)) in batch.enumerated() {
        memcpy(batchedData.buffer.pointer.advanced(by: i * sampleSize * 4),
               sample.buffer.pointer,
               sampleSize * 4)
        memcpy(batchedLabels.buffer.pointer.advanced(by: i * labelSize * 4),
               label.buffer.pointer,
               labelSize * 4)
    }
    
    return (batchedData, batchedLabels)
}

public enum DataError: Error {
    case emptyBatch
    case invalidShape
}
