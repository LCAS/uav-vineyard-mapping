#!/bin/bash
# Diagnostic script to check Flask app configuration

echo "======================================"
echo "  🔍 GCP Annotator Diagnostic Check"
echo "======================================"
echo

# Current working directory
echo "📍 Current Directory:"
pwd
echo

# Check image folder
echo "📁 Image Folder Check:"
IMAGE_FOLDER="../../images/riseholme/august_2024/39_feet"
ABSOLUTE_PATH=$(cd "$(dirname "$0")" && cd "$IMAGE_FOLDER" && pwd 2>/dev/null)

if [ -n "$ABSOLUTE_PATH" ]; then
    echo "✓ Image folder found: $ABSOLUTE_PATH"
    COUNT=$(find "$ABSOLUTE_PATH" -type f \( -name "*.jpg" -o -name "*.JPG" \) 2>/dev/null | wc -l)
    echo "✓ Images found: $COUNT"
    echo "✓ Sample images:"
    find "$ABSOLUTE_PATH" -type f \( -name "*.jpg" -o -name "*.JPG" \) 2>/dev/null | head -3 | sed 's/^/  - /'
else
    echo "❌ Image folder not found!"
    echo "   Expected path: $IMAGE_FOLDER"
    echo "   Current dir: $(pwd)"
    echo "   Try: ls -la $IMAGE_FOLDER"
fi
echo

# Check Python and Flask
echo "🐍 Python & Flask Check:"
python3 -c "import flask; print('✓ Flask version:', flask.__version__)" 2>/dev/null || echo "❌ Flask not installed"
echo

# Check if port 5005 is available
echo "🔌 Port Check:"
if python3 -c "import socket; s = socket.socket(); s.bind(('', 5005)); s.close()" 2>/dev/null; then
    echo "✓ Port 5005 is available"
else
    echo "⚠️  Port 5005 appears to be in use"
    echo "   Try: lsof -i :5005"
fi
echo

# Show file structure
echo "📂 GCP Calibration Folder Structure:"
ls -la | grep -E "\.py|\.html|\.json|\.sh|\.md"
echo

# Summary
echo "======================================"
echo "✓ All checks completed!"
echo
echo "To start the Flask app:"
echo "  bash run_gcp_annotator.sh"
echo
echo "To access the web interface:"
echo "  http://localhost:5005"
echo
echo "To check Flask app status:"
echo "  curl http://localhost:5005/api/debug"
echo "======================================"
