"""
get_zonal_stats.py is a function that computes zonal statistics (mean) when given raster image and geometry list and returns Pandas dataframe of zonal statistics.
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


Call example: 
    df_zonal_statistics = get_zonal_stats.getstats( raster_image, geometries )

"""

import gdal
import rasterio
import spectral
import numpy as np
import pandas as pd
import zonal_stats_function

def getstats(raster, geometrylist):
    
    """
    Used raster image
    
    """
    # Used multiband raster image
    raster_to_be_used = raster
    
    # Check if the raster image is Envi-file or e.g. GeoTIFF
    splitted_raster_name = raster_to_be_used.split('.')
    file_extension = splitted_raster_name[1]
    
    if file_extension=='tif':
        dataset = gdal.Open(raster_to_be_used)
        number_of_bands = dataset.RasterCount
        RASTER = raster_to_be_used
    else:
        dataset = spectral.open_image(raster_to_be_used)
        dimensio = dataset.shape
        number_of_bands = dimensio[2]
        RASTER = splitted_raster_name[0]  # When we use Envi-file, rasterio and rasterstats require the filename without the extension .hdr
        
    # In order to save time, the used raster image will be converted into an array.
    # One band gets one array. To get all the bands from multilayer raster image,
    # we need to loop over each band and append obtained band array to a final list of arrays.
    
    Array_list = []
    for i in range(number_of_bands):
        i += 1
        with rasterio.open(RASTER) as src:
            # numpy array of the raster image --> useful when computing raster statistics.  (https://pythonhosted.org/rasterstats/manual.html --> Raster Data Sources) 
            Affine = src.transform
            Array = src.read(i)
        Array_list.append(Array)
    
    
    """
    Zonal statistics
    
    """
    geometry = geometrylist

    # Create few lists for data management of the results of zonal statistics function
    zonal_value_list = []
    list_of_nans = [np.nan] * number_of_bands
    i = 0
    for element in geometry:
        geometry_element = geometry[i]
        
        # Check if the geometry is Empty or not. (If geometry is Empty, calculating zonal statistics is not possible)
        empty = geometry_element.IsEmpty()
        if empty == True:
            zonal_value_list.append(list_of_nans)
            
        else:
            GEOMETRY = geometry_element.ExportToJson()  # Convert osgeo.ogr.Geometry into GeoJSON format
            # Windows has problems with path length. Here a problem related to path length (ValueError: stat: path too long for Windows),
            # will occur if the GEOMETRY polygon has more than 620 nodes. --> To go around this problem, we read those geometries as strings into the rasterstats.
            # However, if the GEOMETRY polygon has a huge amount of nodes, e.g. over 1000, the problem cannot be avoided.
            nodes = GEOMETRY.count('.')/2
            if nodes > 620 and nodes < 1000:
                GEOMETRY = str(geometry_element)
                # Call for the function that computes the zonal statistics. 
                # INPUT:   - Geometry
                #          - Raster image (as array, needs also the affine)
                #          - Number of bands in the raster
                # OUTPUT:  - out[0]: List of mean values within the given geometry on each raster layer, i.e. band.,
                #            if the geometry is within the raster boundaries. 
                result = zonal_stats_function.zonalstats(GEOMETRY, Array_list, Affine, number_of_bands)
                
            if nodes > 1000:
                result = -9999
            
            else:
                # If the number of nodes is less than 620, the geometry can be read in GeoJSON format.
                result = zonal_stats_function.zonalstats(GEOMETRY, Array_list, Affine, number_of_bands)
                
            # If no proper zonal statistics value is acquired, we add only NaNs to the list of mean values
            if result == -9999:
                zonal_value_list.append(list_of_nans)
            else:
                zonal_value_list.append(result)
        
        i += 1
    
    # Create column names for the dataframe
    bandlist = []
    for round in range(number_of_bands):
        bnumber = 'B'+str(round+1)
        bandlist.append(bnumber)
    
    # Create a dataframe where a row corresponds to one geometry and one column corresponds to one raster band.
    number_of_rows = len(zonal_value_list)
    df_zonal_values = pd.DataFrame(np.array(zonal_value_list).reshape(number_of_rows,number_of_bands), columns = bandlist)
    
    # Function returns dataframe where the zonal statistics are saved.
    return df_zonal_values