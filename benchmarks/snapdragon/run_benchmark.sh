#!/usr/bin/env bash
# run_benchmark.sh
# Safely pushes models, runs the hardware tests, and cleans up the phone

echo "==> Verifying ADB connection..."
adb devices | grep -w "device" > /dev/null
if [ $? -ne 0 ]; then
    echo "Error: No Android device connected or authorized via ADB."
    echo "Please enable Developer Options -> USB Debugging and authorize this laptop."
    exit 1
fi

echo "==> Creating isolated tmp directories on Snapdragon..."
adb shell mkdir -p /data/local/tmp/fusionml/
adb shell mkdir -p /data/local/tmp/fusionml_results/

echo "==> Deploying pipeline binary and TFLite models..."
adb push fusionml_benchmark /data/local/tmp/fusionml/
adb shell chmod +x /data/local/tmp/fusionml/fusionml_benchmark

# Push models built by export_tflite.py
adb push models/*.tflite /data/local/tmp/fusionml/ || {
    echo "Models not found! Run python export_tflite.py first."
    exit 1
}

echo ""
echo "========================================================="
echo "   BEGINNING PHYSICAL HARDWARE TEST ON SNAPDRAGON"
echo "========================================================="
echo " This runs our C++ multi-threaded pipeline directly on"
echo " the Android device. This will take ~10-15 minutes."
echo ""

adb shell "cd /data/local/tmp/fusionml && ./fusionml_benchmark"

echo ""
echo "==> Test Complete! Pulling results back to laptop..."
mkdir -p snapdragon_results
adb pull /data/local/tmp/fusionml_results/results.json ./snapdragon_results/

echo "==> SECURE CLEANUP: Wiping all traces from Uncle's phone..."
adb shell rm -rf /data/local/tmp/fusionml/
adb shell rm -rf /data/local/tmp/fusionml_results/

echo "Done! The phone is totally clean. See ./snapdragon_results/results.json"
