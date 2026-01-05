import XCTest
@testable import FusionML

final class FusionMLTests: XCTestCase {
    
    override func setUp() {
        Fusion.initialize()
    }
    
    // MARK: - Tensor Tests
    
    func testTensorCreation() throws {
        let t = try Fusion.zeros([2, 3])
        XCTAssertEqual(t.shape, [2, 3])
        XCTAssertEqual(t.count, 6)
    }
    
    func testTensorRandom() throws {
        let t = try Fusion.rand([10])
        XCTAssertEqual(t.count, 10)
    }
    
    // MARK: - Linear Algebra Tests
    
    func testMatmul() throws {
        let a = try Fusion.ones([2, 3])
        let b = try Fusion.ones([3, 4])
        let c = try Fusion.linalg.matmul(a, b)
        
        XCTAssertEqual(c.shape, [2, 4])
        XCTAssertEqual(c.toArray()[0], 3.0, accuracy: 0.001)
    }
    
    // MARK: - Neural Network Tests
    
    func testLinearLayer() throws {
        let layer = try Fusion.nn.linear(10, 5)
        let input = GradTensor(try Fusion.rand([4, 10]), requiresGrad: false)
        let output = try layer.forward(input)
        
        XCTAssertEqual(output.shape, [4, 5])
    }
    
    func testSequential() throws {
        let model = Fusion.nn.sequential(
            try Fusion.nn.linear(10, 5),
            Fusion.nn.relu()
        )
        
        let input = GradTensor(try Fusion.rand([2, 10]), requiresGrad: false)
        let output = try model.forward(input)
        
        XCTAssertEqual(output.shape, [2, 5])
    }
    
    // MARK: - Autograd Tests
    
    func testBackward() throws {
        let x = GradTensor(try Fusion.rand([2, 3]), requiresGrad: true)
        let y = try x.sum()
        try Fusion.autograd.backward(y)
        
        XCTAssertNotNil(x.grad)
    }
}
