"""
main_read_stand_variables.py is a Python script to read stand variables from MV_geopackage provided by Metsakeskus.
Copyright (C) 2018 Eelis Halme

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see http://www.gnu.org/licenses/

******************************************************************************************


The script reads variables only on stand level.

In addition, the script calculates zonal mean reflectances for each stand geometry from the given remote sensing image (only ENVI .hdr images).

User needs to define several inputs and constraints.


"""

import time
import pandas as pd

start_time = time.time()
local_time = time.localtime(time.time())
hours = local_time[3]
minutes = local_time[4]
seconds = local_time[5]
print("Running started at: {:d}:{:d}.{:d}\n".format(hours, minutes, seconds))



# =================================================================================================================================================================================

"""
Inputs
  - remote sensing image
  - geopackage
  - output file name
  - stand variables and other fields from geopackage
  - data type
  - date of data
  - minimum for NDVI, also red and NIR bands
  - buffer size
  - increment to get treestandid from standid
  
"""

# Path to raster image to be used (Only ENVI images)
raster_path = "C:/data/Used_data/20170615_reflectance_mosaic_128b.hdr"

# Path to Forest Centre (Metsakeskus) geopackage
gpkg_path = "C:/data/Used_data/MV_Pirkanmaa.gpkg"

# Name of the result csv-file (Will be written to the current working directory)
output_name = "Output_Targets_and_Features_Dataset.csv"

# List of stand variables and fields 
# Note! "area" and "LAI" are stand variables that should be collected always from the geopackage.
var_list = [ "area", "LAI", "meanheight", "date", "fertilityclass", "maintreespecies",'treespecies']

# Type (1, 2, 3)
Type = 2

# Date (data modeled to match the conditions of this date)
# If no date is preferred, give an empty list (Date = [])
Date = [ "2018-01-01" ]

# Minimum NDVI
# Note! The red and NIR bands depend on the used remote sensing image.
# Hyperspectral AISA image with 128 bands: red='B61' and NIR='B91'
# Sentinel-2 image: red='B4' and NIR='B8'
red_band = "B61"
nir_band = "B91"
min_NDVI = 0.61

# Minimum LAI
min_LAI = 0.86

# Size of the inner buffer (give zero if no buffer is wanted, i.e. buffer_size = 0)
buffer_size = -10

# Minimum area for a stand
min_area = 0.5 # ha

# treestandid_inc: increment to get treestandid from stadid
#  - Different values can be used to ge the different scenarios -- 1000000000, 2000000000 or 3000000000
#  - 2000000000 should get the recent values, 3000000000 is future projection
#  - Corresponds to the types found in the geopackage files (type=1, type=2 and type=3)
treestandid_inc = 2000000000



# =================================================================================================================================================================================

"""
Read all the stand variables

"""

# Collect stand variables listed in "var_list" into Python Pandas Dataframe
print("\nCollecting stand variables...")
import get_stand_variables
df_stand_variables = get_stand_variables.getstandvariables(raster_path, gpkg_path, var_list, treestandid_inc, Type, Date)
print("End.")

# Delete all the rows where effective LAI is below the predefined minimum value
df_stand_variables = df_stand_variables[df_stand_variables.LAI > min_LAI]

# Delete all the rows where surface area is below the predefined minimum area
df_stand_variables = df_stand_variables[df_stand_variables.area > min_area]


# Next we add data about tree species to the dataframe if "treespecies" is listed in the list of variables of interest.
# Main tree species is calculated based on basal area (BA) and leaf area index (LAI)
needed_or_not = var_list.count('treespecies')
if needed_or_not > 0:
    # We are interested only in pine, spruce and broadleaved trees.
    # Output has columns for: standid, pine_BA, pine_LB, pine_LAI, spruce_BA, spruce_LB, spruce_LAI, broadleaved_BA, broadleaved_LB, broadleaved_LAI
    print("\nCollecting tree species data...")
    stand_ID_list = df_stand_variables['standid'].tolist()
    import get_species_specific_data
    df_tree_data = get_species_specific_data.getstandtreespeciesdata(stand_ID_list, treestandid_inc, gpkg_path)
    print("End.")
    
    print("\nCalculating main tree species of the stands...")
    import get_tree_species
    df_main_tree_species = get_tree_species.getmaintreespecies( df_tree_data )
    print("End.")
    
    # Let's merge the dataframe of tree species data to the dataframe of the other stand variables
    df_stand_variables = pd.merge(left=df_stand_variables, right=df_main_tree_species, left_on='standid', right_on='standid')



# =================================================================================================================================================================================

"""
Compute zonal statistics

"""

# Create inner buffers (downsize the stands)
geometries = df_stand_variables['geometry'].tolist()  # geometries at the moment
import get_buffers
geometrylist = get_buffers.getbuffers(geometries, buffer_size)  # create buffers

# Calculate zonal mean reflectances 
print("\nCalculating zonal statistics...")
import get_zonal_stats
df_zonal_mean_reflectances = get_zonal_stats.getstats(raster_path, geometrylist)
print("End.")

# Add standid list to the dataframe of zonal statistics
standids = df_stand_variables['standid'].tolist()
d_stands = {'standid':standids}
df_stands = pd.DataFrame(data=d_stands)
df_zonal_mean_reflectances = df_zonal_mean_reflectances.assign(standid=df_stands.values)

# Merging the dataframes of zonal statistics and stand variables
df_final = pd.merge(left=df_stand_variables, right=df_zonal_mean_reflectances, left_on='standid', right_on='standid')

# Replace the 'geometry' column with the buffer geometries
df_final = df_final.drop('geometry', axis=1) # drop the original geometries
d_geom = {'geometry': geometrylist}
df_geom = pd.DataFrame(data=d_geom)
df_final = df_final.assign(geometry=df_geom.values) # add new buffer geometries

# Delete all the rows where there is No Data within the geometry, i.e. delete all NaNs.
df_final = df_final.dropna()



# =================================================================================================================================================================================

"""
Calculate NDVI values

"""

# Extracting the red and NIR band reflections from the dataframe that has the zonal statistics
red_list = df_final[red_band].tolist()
NIR_list = df_final[nir_band].tolist()

# Call function that calculates NDVI values
print("\nCalculating NDVI values...")
import get_NDVI
ndvi_list = get_NDVI.getndvi( red_list, NIR_list )
print("End.")

# Dataframe for the NDVI values
d_NDVI  = {'NDVI': ndvi_list}
df_NDVI = pd.DataFrame(data=d_NDVI)

# Add NDVI values to the same dataframe
df_final = df_final.assign(NDVI=df_NDVI.values)

# Delete also rows where NDVI is less than the predefined minimun value
df_final = df_final[df_final.NDVI > min_NDVI]



# =================================================================================================================================================================================

"""
Output

"""

# Rearrange the columns, so that mean zonal reflectances are the last values and geometries are the first.
stand_var = list(df_stand_variables)
bandlist = list(df_zonal_mean_reflectances)
bandlist.remove('standid')
stand_var.remove('geometry')
df_final = df_final[['geometry'] + stand_var + ['NDVI'] + bandlist]

import os
currentfolder = os.getcwd()
csvfile_name = os.path.join(currentfolder, output_name)
df_final.to_csv(csvfile_name, encoding='utf-8', index=False)

m, s = divmod((time.time() - start_time), 60)
h, m = divmod(m, 60)
print("\nFinished in:")
print("%dh %02dmin %02dsec " % (h, m, s))