import numpy as np
from fusionml._metal.pipeline_scheduler import PipelineScheduler, build_resnet_block
from fusionml._metal.neural_scheduler import NeuralDeviceScheduler

def test_nds():
    # Pre-train the NDS simulator
    nds = NeuralDeviceScheduler()
    if not nds.is_trained:
        nds.train_from_simulated_traces()

    # Create Pipeline with NDS
    sched = PipelineScheduler(verbose=True, use_nds=True)

    print("\nBuilding dummy ResNet pipeline blocks...")
    blocks = []
    # Just 2 blocks to map
    blocks.extend(build_resnet_block(1, 64, 64, 56, 56))
    blocks.extend(build_resnet_block(2, 64, 128, 56, 56, downsample=True))

    for layer in blocks:
        sched.add_layer(layer)

    print("\nCompiling pipeline with Neural Device Scheduler...")
    # This will trigger compilation and profiling then use NDS
    sched.compile(profile_iters=2)

    print("\nNDS Schedule generated successfully.")

if __name__ == "__main__":
    test_nds()
