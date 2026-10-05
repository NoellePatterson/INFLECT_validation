"""
Validation of INFLECT bankfull identification 

This code compares bankfull outputs from INFLECT with action level stages from NWS. 

Noelle Patterson, USU 
August 2026
"""


import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as stats
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPoint, LineString, Point
from sklearn.linear_model import LinearRegression
import os

action_level_df = pd.read_csv('data_outputs/action_levels.csv')
reach_names = action_level_df['reach_name']

reach_name = reach_names[11]
# for reach_name in reach_names:
print(reach_name)
# breakpoint()
# Create folder for reach in outputs if it doesn't exist. 
output_dir = Path('data_outputs') / reach_name
output_dir.mkdir(parents=True, exist_ok=True)

### Import and prepare data ### 

# Test for Boise first
root_dir = Path('/mnt/c/Users/A02466114/OneDrive - USU/Documents/USU_Research/Bankfull Identification/Data/INFLECT_BM_inputs/{}'.format(reach_name))
dem = rasterio.open(root_dir / 'dem/dem_1m.tif')
# import thalweg connecting cross-sections and gage location
thalweg = gpd.read_file(root_dir / 'Thalweg/TW1.shp')
detrend_xs = gpd.read_file(root_dir / 'Cross_Sections/TW1_xs.shp')
# Calculate elevation of action level

reach_row = action_level_df.loc[
    action_level_df['reach_name'].eq(reach_name)
].squeeze('index')
action_stage = reach_row['action_stage_m']
datum_elev = reach_row['nrldb_vertical_datum_ft'] * 0.3048
action_elev_m = action_stage + datum_elev

# Detrend INFLECT bankfull elev to gage location
gage_lat = action_level_df['latitude_DD'][action_level_df['reach_name'] == reach_name]
gage_lon = action_level_df['longitude_DD'][action_level_df['reach_name'] == reach_name]

# Create a GeoDataFrame with Point geometries
gage_coords_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(gage_lon, gage_lat),
    crs='EPSG:4326'
)

# Set CRS to match thalweg
gage_coords_gdf.to_crs(thalweg.crs, inplace=True)
# Get the geometry of the gage
usgs_gage = gage_coords_gdf['geometry'][0]
# Get the geometry of the top cross-section (should be a LineString)
top_xs = detrend_xs.iloc[0]['geometry']
### Detrend calculation to connect gage elevation to cross-sections ###

# Apply a linear detrend to bring bankfull elev to correct level at the gage. 
elevs = []
for index, row in detrend_xs.iterrows():
    line = gpd.GeoDataFrame({'geometry': [row['geometry']]}, crs=detrend_xs.crs)
    tot_len = line.length
    distances = np.arange(0, tot_len[0], 10)  # distance in units meters
    stations = row['geometry'].interpolate(distances) # specify stations on cross-section based on plotting interval
    stations = gpd.GeoDataFrame(geometry=stations, crs=detrend_xs.crs)
    # Extract z elevation at each station along transect
    current_elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))
    xs_thalweg = min(current_elevs)
    elevs.append(xs_thalweg)

x_vals = list(range(0, len(elevs)))
x = np.array(x_vals).reshape(-1, 1)
model = LinearRegression().fit(x, elevs)
slope = model.coef_
intercept = model.intercept_

fit_slope =  slope*x
fit_slope = [val[0] for val in fit_slope]
total_slope = max(fit_slope) - min(fit_slope) # across entire thalweg line, elevation drops this many meters

# Plot sampled elevations and fitted slope along the thalweg
fit_slope_plot = fit_slope + intercept
plt.figure(figsize=(10, 6))
plt.plot(x_vals, [e[0] for e in elevs], 'o-', label='Sampled Elevations')
plt.plot(x_vals, fit_slope_plot, '--', label='Fitted Slope (Linear Trend)')
plt.xlabel('Sample Index Along Thalweg')
plt.ylabel('Elevation (m)')
plt.title('Sampled Elevations and Fitted Slope Along Thalweg, {}'.format(reach_name))
plt.legend()
plt.tight_layout()
plt.savefig('data_outputs/{}/gage_detrending.jpeg'.format(reach_name))

# Calculate total length of thalweg (in meters)
thalweg_tot_len = thalweg.length.sum()

# Find intersection point between top cross-section and thalweg
thalweg_line = thalweg.geometry.iloc[0]  # Assuming single LineString
intersection = detrend_xs.iloc[0].geometry.intersection(thalweg_line)
intersection_gdf = gpd.GeoDataFrame(
    {'reach_name': [reach_name]},
    geometry=[intersection],
    crs=thalweg.crs,
)
intersection_gdf.to_file(output_dir / 'intersection.shp')

# Find the closest point on the thalweg to the gage
projected_dist = thalweg_line.project(usgs_gage)
gage_thalweg_pt = thalweg_line.interpolate(projected_dist)

# Calculate the distance along the thalweg from gage_thalweg to intersection
dist_closest = thalweg_line.project(gage_thalweg_pt)
dist_intersection = thalweg_line.project(intersection)
thalweg_dist = abs(dist_intersection - dist_closest)
detrend_m = total_slope * (thalweg_dist/thalweg_tot_len)
    
# convert bankfull level at top XS to corresponding level at gage site
# move up one directory into INFLECT outputs to get bankfull level at top cross-section

reach_name_INFLECT = reach_row['reach_name_INFLECT']
inflect_dir = Path(__file__).resolve().parent.parent / "INFLECT"
bankfull_df = pd.read_csv(
    inflect_dir / "data_outputs" / reach_name_INFLECT / "max_inflections.csv"
)

bankfull = bankfull_df['bankfull'].iloc[0]
bankfull_gage = bankfull - detrend_m

# Populate df with new data and bankfull comparisonn
action_level_df.loc[
    action_level_df['reach_name'] == reach_name, 'INFLECT_bankfull_gage'
] = bankfull_gage

action_level_df.loc[
    action_level_df['reach_name'] == reach_name, 'NWS_action_elev'
] = action_elev_m

action_level_df.loc[
    action_level_df['reach_name'] == reach_name, 'bankfull_diff'
] = action_elev_m - bankfull_gage

action_level_df.to_csv('data_outputs/action_levels.csv')
# Turn whole thing into a loop through the three reaches
