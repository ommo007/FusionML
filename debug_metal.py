import sys
import platform

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")

try:
    import objc
    print(f"PyObjC version: {objc.__version__}")
except ImportError as e:
    print(f"Failed to import objc: {e}")

try:
    import Metal
    print("SUCCESS: Imported Metal")
    dev = Metal.MTLCreateSystemDefaultDevice()
    print(f"Default Device: {dev.name() if dev else 'None'}")
except ImportError as e:
    print(f"ERROR: Failed to import Metal: {e}")
except Exception as e:
    print(f"ERROR: Exception usage Metal: {e}")
