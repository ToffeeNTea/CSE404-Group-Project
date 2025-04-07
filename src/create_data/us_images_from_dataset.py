import os
import csv
import shutil

from shapely.geometry import Point
import geopandas as gpd
from tqdm import tqdm

# Load a shapefile of US states.
# Update the path below to point to your US states shapefile.
# For example, a shapefile from the U.S. Census Bureau.
states_shapefile = r"D:\\C-Drive-symLinks\\Downloads\\shapefolder\\s_18mr25.shp"  
states = gpd.read_file(states_shapefile)

def isInUS(lat: float, lon: float) -> bool:
    """
    Check if the latitude and longitude are within any US state polygon.
    """
    point = Point(lon, lat)  # Note: Point(x, y) where x=longitude, y=latitude
    # Check if the point lies in any state polygon
    return states.contains(point).any()

def getState(lat: float, lon: float) -> str:
    """
    Given a latitude and longitude, return the name of the US state if the point is in one.
    If the point is not in any state, return None.
    """
    point = Point(lon, lat)
    # Iterate over the states to see which one contains the point.
    for idx, row in states.iterrows():
        if row['geometry'].contains(point):
            return row['NAME']
    return None

def main(image_dir: str, csv_path: str, output_dir: str):
    """
    Process the CSV and images:
      - For each entry in the CSV, extract the image file name, latitude, and longitude.
      - If the coordinates are in the US, determine which state they belong to.
      - Copy the corresponding image into a folder named for that state.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The csv path {csv_path} does not exist.")

    # This dictionary will hold lists of image file names for each state.
    state_images = {}

    # Open and read the CSV file.
    with open(csv_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # first row iheader skip
        # image_filename, lat, lon, date idc about date
        for row in tqdm(csvreader, desc="Processing CSV", unit="row"):
            try:
                image_filename = row[0]
                lat = float(row[1])
                lon = float(row[2])
            except Exception as e:
                print(f"Error processing row {row}: {e}")
                continue

            # if in us, get the state and add to the dictionary
            if isInUS(lat, lon):
                state = getState(lat, lon)
                if state:
                    # Create a new list for this state if it doesn't exist.
                    if state not in state_images:
                        state_images[state] = []
                    state_images[state].append(image_filename)
                else:
                    print(f"Point ({lat}, {lon}) is in US but did not match any state.")
            else:
                pass

    # For each state, create a directory and copy the corresponding images.
    for state, images in tqdm(state_images.items(), desc="Copying images", unit="state"):
        state_dir = os.path.join(output_dir, state)
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)
        for image_filename in images:
            src_image_path = os.path.join(image_dir, image_filename) + '.jpeg'
            dst_image_path = os.path.join(state_dir, image_filename) + '.jpeg'
            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dst_image_path)
            else:
                print(f"Image not found: {src_image_path}")

if __name__ == "__main__":
    image_dir = r"D:\\C-Drive-symLinks\\Downloads\\archive (1)\\images"
    csv_path = r"D:\\C-Drive-symLinks\\Downloads\\archive (1)\\images.csv"
    output_dir = r"D:\\C-Drive-symLinks\\Downloads\\archive (1)\\sorted_by_state"
    
    main(image_dir, csv_path, output_dir)
