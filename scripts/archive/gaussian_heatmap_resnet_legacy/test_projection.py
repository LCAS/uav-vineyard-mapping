#!/usr/bin/env python3
import json
import os
import numpy as np
from pyproj import Transformer

# Test one projection
exif = {
    "lat": 51.597492914,
    "lon": -0.979063143002778,
    "alt": 353.5242967,
    "yaw": 24.9053643724696,
    "pitch": 1.81895001797914,
    "roll": -1.91898026890169,
    "width": 7952,
    "height": 5304
}

# Sample post coordinates from GeoJSON
post_lat = 51.5972744731314
post_lon = -0.978447637406903
post_alt = 353.5

# Test UTM conversion
utm_zone = int((exif['lon'] + 180) / 6) + 1
crs_utm = f"EPSG:326{utm_zone}" if exif['lat'] >= 0 else f"EPSG:327{utm_zone}"
print(f"UTM Zone: {utm_zone}, CRS: {crs_utm}")

transformer = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)

cam_e, cam_n = transformer.transform(exif['lon'], exif['lat'])
post_e, post_n = transformer.transform(post_lon, post_lat)

print(f"Camera: E={cam_e}, N={cam_n}")
print(f"Post: E={post_e}, N={post_n}")

# Vector from camera to post
dE = post_e - cam_e
dN = post_n - cam_n
dU = post_alt - exif['alt']

print(f"Delta: dE={dE}, dN={dN}, dU={dU}")

# NED conversion
P_ned = np.array([dN, dE, -dU])
print(f"P_NED: {P_ned}")
print(f"Distance: {np.linalg.norm(P_ned):.2f} m")
