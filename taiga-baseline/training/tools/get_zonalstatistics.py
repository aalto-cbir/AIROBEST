"""
get_zonalstatistics.py is a function that computes zonal majority from raster's first band.
Copyright (C) 2019 Eelis Halme

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see http://www.gnu.org/licenses/

******************************************************************************************

INPUTS
  - raster image
  - list of OGR geometries
    
OUTPUT
  - Pandas DataFrame including zonal majority values.


"""

def getzonalstatistics(raster, geometrylist):
    
    import spectral
    import rasterio
    import numpy as np
    import pandas as pd
    import zonal_majority
    
    
    """
    Convert the raster image into an array
    
    """
    # Check if the raster image is ENVI .hdr
    splitted_raster_name = raster.split('.')
    file_extension = splitted_raster_name[1]
    
    if file_extension=='hdr':
        dataset = spectral.open_image(raster)
        dimensio = dataset.shape
        number_of_bands = dimensio[2]
        RASTER = splitted_raster_name[0]  # When we use Envi-file, rasterio and rasterstats require the filename without the extension .hdr
    else:
        import sys
        sys.exit("Incorrect raster format! Provide ENVI .hdr raster image.")
        # ENVI .hdr image can be converted from GeoTIFF, for example, using GDAL commands:
        # gdalwarp -overwrite -of ENVI raster_geotiff.tif raster_envihdr -s_srs EPSG:3067 -t_srs EPSG:3067
    
    # As we only check if the Hila geometry has data inside it, 
    # there is no need for every band in the multiband image. We use only one band.
    number_of_bands = 1
    Array_list = []
    for i in range(number_of_bands):
        i += 1
        with rasterio.open(RASTER) as src:
            # numpy array of the raster image --> useful when computing raster statistics.  (https://pythonhosted.org/rasterstats/manual.html --> Raster Data Sources) 
            Affine = src.transform
            Array = src.read(i)
        Array_list.append(Array)
    
    
    """
    Compute zonal majority
    
    """
    # Create few lists for data management of the results of zonal statistics function
    zonal_value_list = []
    list_of_nans = [np.nan] * number_of_bands
    for i in range(len(geometrylist)):
        geometry_element = geometrylist[i]
        
        # Check if the geometry is Empty or not. (If geometry is Empty, calculating zonal statistics is not possible)
        empty = geometry_element.IsEmpty()
        if empty == True:
            zonal_value_list.append(list_of_nans)
        else:
            GEOMETRY = geometry_element.ExportToJson()  # Convert osgeo.ogr.Geometry into GeoJSON format
            
            # Call function that computes the zonal statistics
            result = zonal_majority.zonalmajority(GEOMETRY, Array_list, Affine, number_of_bands)
            
            # If no proper zonal statistics value is acquired, we add only NaNs to the list of mean values
            if result == -9999:
                zonal_value_list.append(list_of_nans)
            else:
                zonal_value_list.append(result)
    
    # Create column names for the dataframe
    bandlist = []
    for round in range(number_of_bands):
        bnumber = 'B'+str(round+1)
        bandlist.append(bnumber)
    
    # Create a dataframe where a row corresponds to one geometry and one column corresponds to one raster band.
    number_of_rows = len(zonal_value_list)
    df_zonal_majority_values = pd.DataFrame(np.array(zonal_value_list).reshape(number_of_rows,number_of_bands), columns = bandlist)
    
    return df_zonal_majority_values