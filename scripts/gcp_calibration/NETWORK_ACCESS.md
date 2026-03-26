# Network Access Guide

## 🌐 Accessing Flask App from Other Machines

The GCP Image Annotator can now be accessed from any machine on your network.

### Prerequisites

1. **Both machines on same network** - Connected to the same WiFi or network
2. **Port 5005 accessible** - No firewall blocking port 5005
3. **Flask app running** - See below for startup instructions

### 📍 Find Your Machine IP

#### Method 1: Auto-detect (Linux/Mac)
```bash
bash get_ip.sh
```

Output example:
```
✅ Your Machine IP: 192.168.1.100
📱 Access from other machines:
   http://192.168.1.100:5005
```

#### Method 2: Manual Detection

**Linux:**
```bash
hostname -I
# or
ifconfig | grep "inet "
```

**Mac:**
```bash
ifconfig | grep "inet "
```

**Windows:**
```cmd
ipconfig
```

Look for an IP address like `192.168.x.x` or `10.0.x.x` (NOT 127.0.0.1)

### 🚀 Start the Flask App

```bash
bash run_gcp_annotator.sh
```

You'll see output like:
```
🌐 Starting Flask server...
   Local:   http://localhost:5005
   Network: http://192.168.1.100:5005  🌍
```

### 💻 Accessing from Other Machines

#### Same Machine
```
http://localhost:5005
```

#### Other Machine on Network
```
http://192.168.1.100:5005
```
(Replace `192.168.1.100` with your actual IP)

### 🔒 Security Notes

⚠️ **Important:**
- The Flask app runs with `debug=True` to allow hot-reloading
- This is fine for local network use
- **Do NOT** expose this to the internet
- No authentication is configured

### ❓ Troubleshooting

#### "Cannot connect from other machine"
1. **Verify IP address** - Run `bash get_ip.sh` again
2. **Check firewall** - May need to allow port 5005
   ```bash
   # Linux firewall
   sudo ufw allow 5005
   ```
3. **Verify network** - Both machines on same network
   ```bash
   # Ping other machine's IP
   ping 192.168.1.100
   ```
4. **Check Flask output** - Should show both local and network URLs

#### "Connection refused"
- Flask app may not be running - check terminal where you started it
- Port 5005 may be in use - try a different port in the script

#### "Connection timeout"
- Firewall may be blocking the connection
- Try accessing from another machine first to verify

### 📱 Mobile/Tablet Access

The web UI works on mobile browsers too!

```
http://192.168.1.100:5005
```

**Note:** Touch interactions may differ from mouse - test before production use.

### 🛠️ Advanced: Change Port

Edit `gcp_image_annotator.py` and change:
```python
app.run(host='0.0.0.0', debug=True, port=5005)  # Change 5005 to desired port
```

Then access via: `http://192.168.1.100:<new-port>`

### 📊 Example Network Setup

```
┌─────────────────────────────────────┐
│      Your Network (192.168.1.x)     │
├─────────────────────────────────────┤
│                                     │
│  Machine A (Host)                   │
│  IP: 192.168.1.100                  │
│  Running Flask at port 5005 ✓       │
│                                     │
│         ↕️  (HTTP requests)          │
│                                     │
│  Machine B (Client)                 │
│  IP: 192.168.1.150                  │
│  Browser: http://192.168.1.100:5005 │
│                                     │
└─────────────────────────────────────┘
```

### 🎯 Complete Workflow

1. **Start Flask** on machine A:
   ```bash
   bash run_gcp_annotator.sh
   ```

2. **Note the network IP** from output (e.g., `192.168.1.100`)

3. **On machine B**, open browser and go to:
   ```
   http://192.168.1.100:5005
   ```

4. **Annotate images** from machine B

5. **Export GCPs** - Files save on machine A

### 📝 Example Session

**Terminal on Machine A:**
```
$ bash run_gcp_annotator.sh
📸 Scanning for images in folder...
✅ Found 42 images

🌐 Finding your machine IP address...
🚀 Starting Flask app...
   Local:   http://localhost:5005
   Network: http://192.168.1.156:5005  🌍

* Running on http://0.0.0.0:5005
* Debug mode: on
```

**Browser on Machine B:**
```
URL: http://192.168.1.156:5005

✅ Connected to Flask app
📸 42 images loaded
🎯 Ready to annotate
```

---

Need help? Check the main [README.md](README.md) or check individual guides:
- [GCP_ANNOTATOR_README.md](GCP_ANNOTATOR_README.md) - Web UI guide
- [GCP_CALIBRATION_README.md](GCP_CALIBRATION_README.md) - Calibration guide
