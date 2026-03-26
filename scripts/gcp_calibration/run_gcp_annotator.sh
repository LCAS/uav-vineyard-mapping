#!/bin/bash
# Quick start script for GCP Annotator

echo "======================================"
echo "  🎯 GCP Image Annotator - Quick Start"
echo "======================================"
echo

# Check if Flask is installed
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing dependencies..."
    pip install -r gcp_annotator_requirements.txt
    echo
fi

# Check if output folder exists
if [ ! -d "gcp_annotations" ]; then
    echo "📁 Creating output folder: gcp_annotations/"
    mkdir -p gcp_annotations
fi

# List images found
echo "📸 Scanning for images in folder..."
IMAGE_COUNT=$(find ../../../images/riseholme/august_2024/39_feet -type f \( -name "*.jpg" -o -name "*.JPG" -o -name "*.png" -o -name "*.PNG" \) 2>/dev/null | wc -l)
echo "✅ Found $IMAGE_COUNT images"
echo

# Find and display IP address
echo "🌐 Finding your machine IP address..."
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$IP" ]; then
    IP=$(ifconfig 2>/dev/null | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
fi
if [ -z "$IP" ]; then
    IP=$(ip addr show 2>/dev/null | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | cut -d'/' -f1 | head -1)
fi

echo "🚀 Starting Flask app..."
if [ -n "$IP" ]; then
    echo "   Local:   http://localhost:5005"
    echo "   Network: http://$IP:5005  🌍"
else
    echo "   http://localhost:5005"
fi
echo
echo "📝 Instructions:"
echo "   1. Open the URL in your browser (see above)"
echo "   2. Click on image to mark control points"
echo "   3. Select which control point (1-4) or None"
echo "   4. Use Previous/Next to navigate"
echo "   5. Export when done"
echo
echo "⌨️  Keyboard:"
echo "   Arrow Left  - Previous image"
echo "   Arrow Right - Next image"
echo
echo "======================================"
echo

python3 gcp_image_annotator.py
