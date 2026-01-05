// Fusion.ane - Neural Engine Backend Module
// Direct ANE operations via Core ML

import Foundation
import CoreML

extension Fusion {
    
    /// ANE (Apple Neural Engine) backend
    /// Best for: conv2d, layer normalization, typical neural network inference
    public enum ane {
        
        // MARK: - Status
        
        /// Check if ANE is available
        public static var isAvailable: Bool {
            // ANE is available on all Apple Fusion Macs
            #if arch(arm64)
            return true
            #else
            return false
            #endif
        }
        
        /// ANE compute unit preference
        public static var computeUnits: MLComputeUnits {
            .cpuAndNeuralEngine
        }
        
        // MARK: - Operations (via Core ML)
        
        /// Run model on ANE
        /// Note: For best ANE utilization, use pre-compiled Core ML models
        public static func predict(model: MLModel, input: MLFeatureProvider) throws -> MLFeatureProvider {
            try model.prediction(from: input)
        }
        
        /// Load model configured for ANE
        public static func loadModel(at url: URL) throws -> MLModel {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            
            // Compile if needed
            let compiledURL = try MLModel.compileModel(at: url)
            return try MLModel(contentsOf: compiledURL, configuration: config)
        }
        
        // MARK: - Recommended Operations for ANE
        
        /// Operations that run well on ANE:
        /// - Convolution (conv1d, conv2d)
        /// - Batch normalization
        /// - Layer normalization  
        /// - Standard neural network inference
        ///
        /// Operations that are slow on ANE:
        /// - Raw matrix multiplication (use gpu or cpu instead)
        /// - Custom operations
        
        /// Note about ANE usage
        public static let notes = """
        ANE (Neural Engine) is optimized for:
        ✓ Convolution operations
        ✓ Normalization layers
        ✓ Pre-compiled Core ML models
        
        For raw matmul, use Fusion.linalg.matmul (GPU+CPU split)
        or Fusion.gpu.matmul / Fusion.cpu.matmul directly.
        """
    }
}
