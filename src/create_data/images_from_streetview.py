# https://download.geonames.org/export/dump/

import csv
import requests
import time
from dotenv import load_dotenv
from pathlib import Path
import os
#from tqdm import tqdm

load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if GOOGLE_MAPS_API_KEY is None:
    raise ValueError("No API key found. Please set GOOGLE_MAPS_API_KEY in your .env file.")

# This variable is importatnt, it determines how many lines of the csv file will be read
# which directly relates to how many images will be downloaded (up to 4 per valid location)
LINES_TO_READ = 5

# Define a list of headings (in degrees) you want to check.
headings = [0, 90, 180, 270]

# Input and output CSV file names
PARENT_FOLDER = Path(__file__).resolve().parent.parent.parent

# Size of the images
IMAGE_SIZE = "640x640"

def download_images_for_location(city, lat, lon):
    """
    For a given city location (lat, lon), check multiple headings using the
    Street View metadata endpoint. For each heading where an image exists,
    download the image using the Street View image API and save it to OUTPUT_DIR.
    Returns the count of images downloaded for this location.
    """
    image_count = 0
    # Create a filename-safe version of the city name
    city_safe = city.replace(" ", "_")
    
    for heading in headings:
        # First, check if an image exists for this heading
        meta_url = (
            f"https://maps.googleapis.com/maps/api/streetview/metadata"
            f"?location={lat},{lon}&heading={heading}&key={GOOGLE_MAPS_API_KEY}"
        )
        meta_response = requests.get(meta_url)
        meta_data = meta_response.json()
        
        if meta_data.get("status") == "OK":
            # Build the image URL with the desired size and parameters
            image_url = (
                f"https://maps.googleapis.com/maps/api/streetview"
                f"?size={IMAGE_SIZE}&location={lat},{lon}&heading={heading}&key={GOOGLE_MAPS_API_KEY}"
            )
            image_response = requests.get(image_url)
            
            if image_response.status_code == 200:
                file_name = f"{city_safe}_{heading}_({lat}_{lon}).jpg"
                file_path = os.path.join(PARENT_FOLDER / 'database' / 'image_data', file_name)
                with open(file_path, "wb") as f:
                    f.write(image_response.content)
                #print(f"Downloaded image for {city} at heading {heading}.")
                image_count += 1
            else:
                print(f"Failed to download image for {city} at heading {heading}.")
            
            # Brief pause to respect rate limits
            time.sleep(0.1)
        else:
            print(f"No image available for {city} at heading {heading}.")
            break
    
    return image_count
    


def main():
    with open(PARENT_FOLDER / 'database' / 'csv_data' / 'cities.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        i = 0
        count = 0
        for row in reader:
            i += 1
            if i > LINES_TO_READ:
                break
            city = row.get('name')
            lat = row.get('latitude')
            lon = row.get('longitude')
            count += download_images_for_location(city, lat, lon)

        print(f"Downloaded {count} images.")

if __name__ == "__main__":
   main()


