#!/bin/bash
# Quick helper to find your machine's IP address

echo "======================================"
echo "  🔍 Finding Your Machine IP Address"
echo "======================================"
echo

# Try different methods to get IP
IP=$(hostname -I 2>/dev/null | awk '{print $1}')

if [ -z "$IP" ]; then
    IP=$(ifconfig 2>/dev/null | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
fi

if [ -z "$IP" ]; then
    IP=$(ip addr show 2>/dev/null | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | cut -d'/' -f1 | head -1)
fi

if [ -n "$IP" ]; then
    echo "✅ Your Machine IP: $IP"
    echo
    echo "📱 Access from other machines:"
    echo "   http://$IP:5005"
    echo
    echo "💻 Access from this machine:"
    echo "   http://localhost:5005"
    echo
    echo "======================================"
else
    echo "❌ Could not automatically detect IP address"
    echo
    echo "Manual methods:"
    echo "  Linux/Mac:  hostname -I  or  ifconfig"
    echo "  Windows:    ipconfig"
    echo
    fi
