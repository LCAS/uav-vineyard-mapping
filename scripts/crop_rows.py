# Required libraries
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec

import rasterio
import fiona
from pyproj import Transformer
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks

# Open the GeoTIFF file
imgRaster = rasterio.open('../images/100_feet/odm_orthophoto.tif')
im = imgRaster.read(1)  # Read the first band

# Display the image
fig = plt.figure(figsize=(12, 8))
plt.imshow(im, cmap='gray')
plt.title('Input Image')
plt.savefig("croprows_input.png", dpi=300)
# plt.show()

# Compute the Canny filter
edges = feature.canny(im, sigma=3)

# Display edges
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(edges, cmap=plt.cm.gray)
ax.set_title(r'Canny filter, $\sigma=3$')
plt.savefig("croprows_canny_.png", dpi=300)
# plt.show()

# Hough Line Transform
precision = 2  # Set a precision of 2 degrees
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, int(180 / precision), endpoint=False)
h, theta, d = hough_line(edges, theta=tested_angles)

# Display the Hough Transform results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.imshow(edges, cmap=cm.gray)
ax1.set_title('Edge Detection')
ax2.imshow(np.log(1 + h), extent=[np.rad2deg(theta[0]), np.rad2deg(theta[-1]), d[-1], d[0]],
           cmap=cm.YlGn, aspect=1 / 1.5)
ax2.set_title('Hough Transform')
plt.savefig("croprows_hough.png", dpi=300)
# plt.show()

# Extract lines and angles
selDiag = edges.shape[1]  # Width of the image
progRange = range(selDiag)

totalLines = []
angleList = []

for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    if angle in [np.pi/2, -np.pi/2]:
        cols = [prog for prog in progRange]
        rows = [y0 for _ in progRange]
    elif angle == 0:
        cols = [x0 for _ in progRange]
        rows = [prog for prog in progRange]
    else:
        c0 = y0 + x0 / np.tan(angle)
        cols = [prog for prog in progRange]
        rows = [col * np.tan(angle + np.pi / 2) + c0 for col in cols]

    # Collect line points
    line_points = []
    for col, row in zip(cols, rows):
        if 0 <= col < edges.shape[1] and 0 <= row < edges.shape[0]:
            line_points.append((row, col))
    totalLines.append(line_points)

    # Store angle
    if math.degrees(angle + np.pi / 2) > 90:
        angleList.append(180 - math.degrees(angle + np.pi / 2))
    else:
        angleList.append(math.degrees(angle + np.pi / 2))

# Reproject to Latitude and Longitude
raster_crs = imgRaster.crs
transformer = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)

totalLinesLatLon = []
for line in totalLines:
    lineLatLon = []
    for row, col in line:
        easting, northing = imgRaster.xy(row, col)  # Pixel to projected coordinates
        lon, lat = transformer.transform(easting, northing)  # Projected to lat/lon
        lineLatLon.append((lon, lat))
    totalLinesLatLon.append(lineLatLon)

# Write the GeoJSON file
schema = {
    'geometry': 'LineString',
    'properties': {'angle': 'float'}
}

with fiona.open('croprows_latlon.geojson', mode='w', driver='GeoJSON',
                schema=schema, crs="EPSG:4326") as outJson:
    for index, line in enumerate(totalLinesLatLon):
        feature = {
            'geometry': {'type': 'LineString', 'coordinates': line},
            'properties': {'angle': angleList[index]}
        }
        outJson.write(feature)

# Display the final lines on the image
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(im, cmap='gray')
ax.set_title('Detected Lines Over Original Image')

for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    ax.axline((x0, y0), slope=np.tan(angle + np.pi / 2), color='red', linewidth=2)

plt.savefig("croprows_final.png", dpi=300)
# plt.show()
