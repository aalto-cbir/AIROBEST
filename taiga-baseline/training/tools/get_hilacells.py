"""
get_hilacells.py is a function that collects Hila grid cells within the boundaries of the used raster image.
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
  - Path to Hila grid cell geopackage
  - Raster image (ENVI .hdr image)
  
OUTPUTS
  - Pandas DataFrame including Hila data within the raster image.



"""

import rasterio
import pandas as pd
from osgeo import ogr
from hypdatatools_img import *
from hypdatatools_gdal import *

def gethilacells(Hila_data, Raster_image):
    
    # If sometime needed, more hila variables can be collected using this list
    var_list = []
    
    # Needed fields and other important fields
    needed_fields = var_list + ["gridcellid"]
    
    # Field names from table "gridcell"
    fields_gridcell = geopackage_getfieldnames( Hila_data, "gridcell" )
    
    # Field names that correspond to the field names of interest
    fields_hila = list(set(fields_gridcell).intersection(needed_fields))
    
    # Read all data from this table into a Python Pandas DataFrame
    fields = ["id", "geometry"] + fields_hila
    out_hila = vector_getfeatures( Hila_data, fields )
    del out_hila[2:4] # delete two unncessary NoneType lists

    # Create a dataframe where all the data of fields
    keys = fields # field names
    values = out_hila  # field data
    dictionary = dict(zip(keys, values))
    df_hila = pd.DataFrame.from_dict(dictionary) # dataframe including the entire table 'stand'
    
    
    # --------------------------------------------------------------------------------------------------------------
    
    
    # Choose only those hila grid cells that are within the boundaries of the used remote sensing image
    # First read raster corner points and create a WKT polygon from these
    splitted_raster_name = Raster_image.split('.')
    file_extension = splitted_raster_name[1]
    if file_extension == 'hdr':
        RASTER = splitted_raster_name[0]
    else:
        import sys
        sys.exit("Incorrect raster format! Provide ENVI .hdr raster image.")
        # ENVI .hdr image can be converted from GeoTIFF, for example, using GDAL commands:
        # gdalwarp -overwrite -of ENVI raster_geotiff.tif raster_envihdr -s_srs EPSG:3067 -t_srs EPSG:3067
    raster_img = rasterio.open(RASTER)
    Bounds = raster_img.bounds # Returns list of 4 elements: [0]left X, [1]lower Y, [2]right X, [3]upper Y
    
    # Create a geometry around the raster
    shape = ogr.Geometry(ogr.wkbLinearRing)
    shape.AddPoint(Bounds[0],Bounds[3])
    shape.AddPoint(Bounds[2],Bounds[3])
    shape.AddPoint(Bounds[2],Bounds[1])
    shape.AddPoint(Bounds[0],Bounds[1])
    shape.AddPoint(Bounds[0],Bounds[3]) # Add the first node also here. This eventually connects all the nodes.
    raster_bounds = ogr.Geometry(ogr.wkbPolygon) # Create the final geometry
    raster_bounds.AddGeometry(shape)
    
    # Check which hila grid cells are completely within the raster bounds
    hilageometries = df_hila['geometry'].tolist()
    gridcellids = df_hila['gridcellid'].tolist()
    gridcellid_list = []
    for k in range(len(hilageometries)):
        Check = raster_bounds.Contains(hilageometries[k])
        if Check == True:
            gridcellid_list.append(gridcellids[k])
    
    # Constructing another Python Pandas DataFrame from only the standids that we need
    d_hilacells = {'gridcellid': gridcellid_list}
    df_needed_hilacells = pd.DataFrame(data=d_hilacells)
    
    # Merge with the earlier one
    dataframe = pd.merge(left=df_needed_hilacells, right=df_hila, left_on='gridcellid', right_on='gridcellid')
    
    return dataframe