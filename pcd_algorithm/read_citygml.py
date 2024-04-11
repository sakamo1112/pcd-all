import json
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def visualize_geojson(geojson):
    """
    Visualize a GeoJSON file using GeoPandas and Matplotlib.
    
    Parameters:
    - geojson: The GeoJSON data to visualize.
    """
    # Convert GeoJSON to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(geojson["features"])
    
    # Plot the GeoDataFrame
    gdf.plot()
    plt.show()

def convert_gdf_to_geojson(gdf):
    """
    Convert a GeoDataFrame to a GeoJSON format.
    
    Parameters:
    - gdf: The GeoDataFrame to convert.
    
    Returns:
    - A GeoJSON representation of the GeoDataFrame.
    """
    return json.loads(gdf.to_json())

def paint_buildings_by_height(gdf, height_threshold=50):
    """
    Paint buildings in a GeoDataFrame based on their height.
    
    Parameters:
    - gdf: The GeoDataFrame containing building data.
    - height_threshold: The height threshold for painting buildings.
    """
    # Create a new column 'color' in the GeoDataFrame
    gdf["color"] = "blue"  # Default color
    gdf.loc[gdf["measuredHeight"] >= height_threshold, "color"] = "red"  # Set color to red for buildings taller than 50m
    
    # Plot the GeoDataFrame with colors based on the 'color' column
    fig, ax = plt.subplots()
    for color in gdf["color"].unique():
        gdf[gdf["color"] == color].plot(ax=ax, color=color)
    
    plt.show()

if __name__ == "__main__":
    gdf = gpd.read_file("data/plateau_citygml/udx/bldg/53394548_bldg_6697_op.gml")
    #gdf = gdf.head(500)
    # print(gdf["measuredHeight"])
    paint_buildings_by_height(gdf)
    
    plt.show()



