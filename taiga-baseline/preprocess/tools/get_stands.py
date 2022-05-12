"""
get_stands.py is a function that reads the required stands from the used geopackage.
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
  - path to stand geopackage
  - csv-list of required stands (list of stand IDs)
 
OUTPUT
  - Pandas DataFrame including stand geometries
    that have been downsized by 10 meters.


"""

import pandas as pd
from hypdatatools_img import *
from hypdatatools_gdal import *

def getstands(Stand_data, Stand_list):
    
    # If sometime needed, more stand variables can be collected using this list
    var_list = []
    
    # Needed fields and other important fields
    needed_fields = var_list + ["standid", "type", "treestandid", "geometry"]
    
    # All field names from table 'stand'
    fields_stand = geopackage_getfieldnames( Stand_data, 'stand' )
    
    # Field names that correspond to the field names of interest
    fields_from_stand = list(set(fields_stand).intersection(needed_fields))
    
    # Read all data from this table into a Python Pandas DataFrame
    out_stand = vector_getfeatures( Stand_data, fields_stand )
    del out_stand[2:4] # delete two unncessary NoneType lists

    # Create a dataframe where all the data of fields from table 'stand' are collected
    keys = fields_stand # field names
    values = out_stand  # field data
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame.from_dict(dictionary) # dataframe including the entire table 'stand'
    
    # Choose only those columns from the dataframe that are given as function inputs or are otherwise needed
    df = df[fields_from_stand]
    
    # Using the list of required stands, we filter out the unnecessary stands
    df_stands = pd.read_csv( Stand_list, encoding='utf-8' ) # standid list
    dataframe = pd.merge(left=df, right=df_stands, left_on='standid', right_on='standid') # merge using only the required stands
    
    # Downsize the forest stands
    buffer_size = -10  # in meters
    orig_geoms = dataframe['geometry'].tolist()
    import get_buffers
    new_geoms  = get_buffers.getbuffers( orig_geoms, buffer_size )
    
    # Replace original geometries in the dataframe with the new downsized geometries
    dataframe = dataframe.drop('geometry', axis=1) # drop the original geometries
    df_geom = pd.DataFrame(data={'geometry': new_geoms})
    dataframe = dataframe.assign(geometry=df_geom.values) # add new buffer geometries
    
    return dataframe