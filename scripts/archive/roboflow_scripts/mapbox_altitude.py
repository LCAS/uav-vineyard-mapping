import requests
from PIL import Image
from io import BytesIO
import math
import json

# Load the Mapbox Access Token
with open("../../config/api_key.json", 'r') as file:
    config = json.load(file)
MAPBOX_ACCESS_TOKEN = config.get("MAPBOX_API_KEY")

# Function to convert Lat/Lon to XYZ tile coordinates
def latlon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(math.radians(lat)) + (1 / math.cos(math.radians(lat)))) / math.pi) / 2.0 * n)
    return xtile, ytile

# Function to fetch elevation from Mapbox Terrain API
def get_elevation(lat, lon, zoom=14):
    xtile, ytile = latlon_to_tile(lat, lon, zoom)
    url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{zoom}/{xtile}/{ytile}@2x.pngraw?access_token={MAPBOX_ACCESS_TOKEN}"
    
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        r, g, b, _ = image.getpixel((0, 0))  # Get RGB values from pixel
        elevation = -10000 + ((r * 256**2 + g * 256 + b) * 0.1)
        return elevation
    else:
        print("Error fetching elevation data")
        return None

if __name__ == "__main__":
    # Example coordinates
    coordinates = [
        (51.597257, -0.978198935),  # Expected: ~236.761m
        (51.59596369, -0.976231233)  # Expected: ~239.834m
    ]

    for lat, lon in coordinates:
        elevation = get_elevation(lat, lon)
        if elevation is not None:
            print(f"Elevation at ({lat}, {lon}): {elevation:.2f} meters")
        else:
            print(f"Failed to fetch elevation for ({lat}, {lon})")
