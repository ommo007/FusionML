#!/usr/bin/env bash
# run_benchmark.sh
# Safely pulls the Cloud-Compiled C++ Android binary, deploys along with models, runs tests, and cleans up

echo "==> Verifying ADB connection..."
adb devices | grep -w "device" > /dev/null
if [ $? -ne 0 ]; then
    echo "Error: No Android device connected or authorized via ADB."
    echo "Please enable Developer Options -> USB Debugging and tap Allow on the phone."
    exit 1
fi

echo "==> Downloading Cloud-Compiled Binary using GitHub CLI (gh)..."
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Install it via: brew install gh"
    exit 1
fi

# Download the latest precompiled artifact from GitHub Actions
rm -rf fusionml_benchmark_binary/
gh run download -n fusionml_benchmark_binary || {
    echo "Failed to download artifact or no successful builds found."
    echo "Check the Actions tab in your GitHub repository."
    exit 1
}
mv fusionml_benchmark_binary/fusionml_benchmark ./fusionml_benchmark
chmod +x fusionml_benchmark

echo "==> Creating isolated tmp directories on Snapdragon..."
adb shell mkdir -p /data/local/tmp/fusionml/
adb shell mkdir -p /data/local/tmp/fusionml_results/

echo "==> Deploying pipeline binary and TFLite models..."
adb push fusionml_benchmark /data/local/tmp/fusionml/
adb shell chmod +x /data/local/tmp/fusionml/fusionml_benchmark

# Push models built by export_tflite.py
adb push models/*.tflite /data/local/tmp/fusionml/ || {
    echo "Models not found! Make sure you ran python export_tflite.py first."
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

echo "==> SECURE CLEANUP: Wiping all traces from Uncle phone..."
adb shell rm -rf /data/local/tmp/fusionml/
adb shell rm -rf /data/local/tmp/fusionml_results/

echo "Done! The phone is totally clean. See ./snapdragon_results/results.json"
