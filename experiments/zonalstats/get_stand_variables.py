"""
get_stand_variables.py function collects stand variables from Finnish Forest Centre (Metsakeskus) geopackage files.
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


INPUT:    
  - raster_path: path to the used raster file (ENVI format .hdr file), only stands that locate within the raster are collected
  - gpkg_path: path to the used geopackage file from Metsakeskus (MV_file.gpkg)
  - var_list: list of stand variables of interest
  - treestandid_inc: increment to get treestandid from standid
      + Different values can be used to ge the different scenarios -- 1000000000, 2000000000 or 3000000000
      + 2000000000 should get the recent values, 3000000000 is future projection, 1000000000 are measurements/inventory
      + Corresponds to the types found in the geopackage files (type=1, type=2 and type=3)
  - Type: 1, 2, 3
  - Date: the dates from which data is wanted to be read. Has to be given as list, even an empty list [].

OUTPUT: 
  - Python Pandas DataFrame with the stand variables that were given as inputs.
  
"""

import rasterio
import numpy as np
import pandas as pd
from osgeo import ogr
from hypdatatools_img import *
from hypdatatools_gdal import *
from shapely.wkt import loads
from shapely.geometry import polygon, mapping

def getstandvariables(raster_path, gpkg_path, var_list, treestandid_inc, Type, Date):
    
    # Used geopackage
    filename_mv = gpkg_path
    
    # Needed fields: forest variables and other important fields
    needed_fields = var_list + ["standid", "type", "treestandid", "geometry"]
    
    
    # =================================================================================================================================================================================
    
    """
    Data from table "stand"
    
    """
    # All field names from table 'stand'
    fields_stand = geopackage_getfieldnames( filename_mv, 'stand' )
    
    # Field names that correspond to the field names of interest
    fields_from_stand = list(set(fields_stand).intersection(needed_fields))
    
    # Read all data from this table into a Python Pandas DataFrame
    out_stand = vector_getfeatures( filename_mv, fields_stand )
    del out_stand[2:4] # delete two unncessary NoneType lists

    # Create a dataframe where all the data of fields from table 'stand' are collected
    keys = fields_stand # field names
    values = out_stand  # field data
    dictionary = dict(zip(keys, values))
    df_table_stand = pd.DataFrame.from_dict(dictionary) # dataframe including the entire table 'stand'
    
    # Which stands are inside the used raster image?
    # The output here == i_stand --> This list tells the indices we need.
    standid_list = df_table_stand['standid'].tolist()
    geom_list = df_table_stand['geometry'].tolist()
    i_stand = geometries_subsetbyraster( geom_list, raster_path, reproject=False )
    
    # These standids and geometries locate within the raster image being used
    Standids = []  # collect standids we want to use with the use of indices
    Geometry_lst = []
    for i in i_stand:
        Standids.append(standid_list[i])
        Geometry_lst.append(geom_list[i])
        
    # The geometries_subsetbyraster has some flaws. Some stands, which actually are beyond the bounds of the raster image, are left to the dataset.
    # These can be deleted by checking if they locate completely within the raster bounds or not.
    # First read raster corner points and create a WKT polygon from these
    splitted_raster_name = raster_path.split('.')
    file_extension = splitted_raster_name[1]
    if file_extension == 'hdr':
        RASTER = splitted_raster_name[0]
    raster_img = rasterio.open(RASTER)
    Bounds = raster_img.bounds # Returns list of 4 elements: [0]left X, [1]lower Y, [2]right X, [3]upper Y
    
    # Create a geometry around the raster
    shape = ogr.Geometry(ogr.wkbLinearRing)
    shape.AddPoint(Bounds[0],Bounds[3])
    shape.AddPoint(Bounds[2],Bounds[3])
    shape.AddPoint(Bounds[2],Bounds[1])
    shape.AddPoint(Bounds[0],Bounds[1])
    shape.AddPoint(Bounds[0],Bounds[3]) # Add the first node also here. This eventually connects all the nodes.
    geometry = ogr.Geometry(ogr.wkbPolygon) # Create the final geometry
    geometry.AddGeometry(shape)
    Geometry_in_WKT = geometry.ExportToWkt() # Final geometry in Well-Known-Text
    
    # Check which stands are completely within the raster bounds
    new_standid_lst = []
    k = 0
    for k in range(len(Geometry_lst)):
        stand_id = Standids[k]
        wkt_geo = (Geometry_lst[k]).ExportToWkt()
        poly1 = loads(Geometry_in_WKT)
        poly2 = loads(wkt_geo)
        Check = poly1.contains(poly2)   # .contains function returns False if polygon was outside or on the border of another polygon
        if Check == True:               # .contains function returns True if polygon is completely within another polygon
            new_standid_lst.append(stand_id)
    
    # Constructing another Python Pandas DataFrame from only the standids that we need
    d_stands = {'standid': new_standid_lst}
    df_needed_stands = pd.DataFrame(data=d_stands)
    
    # Merge the two dataframes: this way we can get rid off all unnecessary stands, which are beyond the borders of this raster that is used.
    df_t_stand = pd.merge(left=df_needed_stands, right=df_table_stand, left_on='standid', right_on='standid')
    
    # Choose only those columns from the final dataframe that are given as function inputs or are otherwise needed
    df_t_stand = df_t_stand[fields_from_stand]
    
    
    # =================================================================================================================================================================================
    
    """
    Data from table "treestand" where 'type' is given as function input
    
    """
    # We open the data table and keep it open. Conn and c can be used in the place of filename_mv until the file is closed.
    conn = sqlite3.connect(filename_mv)
    c = conn.cursor()
    
    # Obtaining data from every field in this table.
    if Type == 1:
        id_treestand_list           = geopackage_getvalues( c, "treestand", "id",           additionalconstraint="where type=1" )
        treestandid_treestand_list  = geopackage_getvalues( c, "treestand", "treestandid",  additionalconstraint="where type=1" )
        standid_treestand_list      = geopackage_getvalues( c, "treestand", "standid",      additionalconstraint="where type=1" )
        date_treestand_list         = geopackage_getvalues( c, "treestand", "date",         additionalconstraint="where type=1" )
        type_treestand_list         = geopackage_getvalues( c, "treestand", "type",         additionalconstraint="where type=1" )
        datasource_treestand_list   = geopackage_getvalues( c, "treestand", "datasource",   additionalconstraint="where type=1" )
        creationtime_treestand_list = geopackage_getvalues( c, "treestand", "creationtime", additionalconstraint="where type=1" )
        updatetime_treestand_list   = geopackage_getvalues( c, "treestand", "updatetime",   additionalconstraint="where type=1" )
    if Type == 2:
        id_treestand_list           = geopackage_getvalues( c, "treestand", "id",           additionalconstraint="where type=2" )
        treestandid_treestand_list  = geopackage_getvalues( c, "treestand", "treestandid",  additionalconstraint="where type=2" )
        standid_treestand_list      = geopackage_getvalues( c, "treestand", "standid",      additionalconstraint="where type=2" )
        date_treestand_list         = geopackage_getvalues( c, "treestand", "date",         additionalconstraint="where type=2" )
        type_treestand_list         = geopackage_getvalues( c, "treestand", "type",         additionalconstraint="where type=2" )
        datasource_treestand_list   = geopackage_getvalues( c, "treestand", "datasource",   additionalconstraint="where type=2" )
        creationtime_treestand_list = geopackage_getvalues( c, "treestand", "creationtime", additionalconstraint="where type=2" )
        updatetime_treestand_list   = geopackage_getvalues( c, "treestand", "updatetime",   additionalconstraint="where type=2" )
    if Type == 3:
        id_treestand_list           = geopackage_getvalues( c, "treestand", "id",           additionalconstraint="where type=3" )
        treestandid_treestand_list  = geopackage_getvalues( c, "treestand", "treestandid",  additionalconstraint="where type=3" )
        standid_treestand_list      = geopackage_getvalues( c, "treestand", "standid",      additionalconstraint="where type=3" )
        date_treestand_list         = geopackage_getvalues( c, "treestand", "date",         additionalconstraint="where type=3" )
        type_treestand_list         = geopackage_getvalues( c, "treestand", "type",         additionalconstraint="where type=3" )
        datasource_treestand_list   = geopackage_getvalues( c, "treestand", "datasource",   additionalconstraint="where type=3" )
        creationtime_treestand_list = geopackage_getvalues( c, "treestand", "creationtime", additionalconstraint="where type=3" )
        updatetime_treestand_list   = geopackage_getvalues( c, "treestand", "updatetime",   additionalconstraint="where type=3" )
    
    # Constructing DataFrame
    d_t_treestand  = {'ID_t_ts': id_treestand_list, 'treestandid': treestandid_treestand_list, 'standid': standid_treestand_list, 'date': date_treestand_list,
           'type': type_treestand_list, 'datasource':datasource_treestand_list, 'creationtime':creationtime_treestand_list, 'updatetime':updatetime_treestand_list}
    df_t_treestand = pd.DataFrame(data=d_t_treestand)
    
    # If some preferred dates are given as function input, let's use only those dates.
    if len(Date) > 0:
        found_dates = date_treestand_list  # the dates the dataframe has at the moment
        corresponding_dates = list(set(found_dates).intersection(Date))  # do we have the preferred dates in the dataframe at the moment
        
        # Collect the unnecessary dates
        copy_df = df_t_treestand
        for d in corresponding_dates:
            copy_df = copy_df[copy_df.date != d ]
        dates_to_delete = list(set(copy_df['date'].tolist()))  # the unnecessary dates
        
        # Delete unnecessary dates from the dataframe
        for day in dates_to_delete:
            df_t_treestand = df_t_treestand[df_t_treestand.date != day ]
    
    # Check which columns are needed and which are unnecessary.
    # All field names from table 'treestand'
    fields_treestand = ["id", "treestandid", "standid", "date", "type", "datasource", "creationtime", "updatetime"]
    
    # Field names that correspond to the field names of interest
    fields_from_treestand = list(set(fields_treestand).intersection(needed_fields))
    
    # Choose only those columns from the dataframe that are necessary
    df_t_treestand = df_t_treestand[fields_from_treestand]
    
    # Merge with the dataframe that has already data from table 'stand'
    df_dataframe = pd.merge(left=df_t_stand, right=df_t_treestand, left_on='standid', right_on='standid')
    
    
    # =================================================================================================================================================================================
    
    """
    Fields from table "treestandsummary"
    
    """
    # All field names in the table 'treestandsummary'
    fields_treestandsummary = geopackage_getfieldnames( filename_mv, 'treestandsummary' )
    
    df_data_standsummary = [] # collect data for the dataframe
    for j in range(len(fields_treestandsummary)):
        read_field = geopackage_getvalues( c, "treestandsummary", fields_treestandsummary[j], additionalconstraint=None )
        df_data_standsummary.append(read_field)  # append list of field data to a larger list
        
    # Create a dataframe where all the data of fields from table 'stand' are collected
    Keys = fields_treestandsummary  # field names
    Values = df_data_standsummary   # field data
    Dictionary = dict(zip(Keys, Values))
    df_table_treestandsummary = pd.DataFrame.from_dict(Dictionary) # dataframe including the entire table 'treestandsummary'
    
    # Close the connection to tables.
    conn.close()
    
    # In the new dataframe including all the data from table 'treestandsummary', treestandids are not in the right order. The correct order is told in the id list of current table.
    # We need to create a new order for 'treestandid' using the 'id' column.
    treestandid_treestandsummary_list = df_table_treestandsummary['treestandid'].tolist()
    table_id_treestandsummarylist = df_table_treestandsummary['id'].tolist()
    
    new_treestandid_list = [np.nan] * (len(treestandid_treestandsummary_list))
    j = 0
    for indexvalue in table_id_treestandsummarylist:
        new_treestandid_list[indexvalue-1] = treestandid_treestandsummary_list[j]
        j += 1
    
    # Create also new ID list
    new_id_list = list(range(1, (len(table_id_treestandsummarylist)+1)))
    
    # Delete the two older columns from the dataframe and add the new lists of 'treestandid' and 'id'
    #  - Delete old
    df_table_treestandsummary = df_table_treestandsummary.drop('treestandid', axis=1)
    df_table_treestandsummary = df_table_treestandsummary.drop('id', axis=1)
    #  - Add new
    d_new = {'id': new_id_list, 'treestandid': new_treestandid_list} # new lists into a dataframe
    df_new = pd.DataFrame(data=d_new)
    df_table_treestandsummary = pd.concat([df_new, df_table_treestandsummary], axis=1) # concatenating the two dataframes
    
    # Check the fields that are needed, i.e. check the fields corresponding to the field names of interest
    fields_from_treestandsummary = list(set(fields_treestandsummary).intersection(needed_fields))
    
    # Choose only those columns from the dataframe that are given as function inputs or are otherwise needed
    df_table_treestandsummary = df_table_treestandsummary[fields_from_treestandsummary]
    
    # Merge with the earlier dataframe that includes other forest variables of interest and other important fields (using: 'treestandid')
    df_final_dataframe = pd.merge(left=df_dataframe, right=df_table_treestandsummary, left_on='treestandid', right_on='treestandid')
    
    
    # =================================================================================================================================================================================
    
    """
    Compute effective LAI (if it is listed in the stand variables of interest)
    
    """
    need_for_LAI = needed_fields.count('LAI') # is LAI listed in the list of variables of interest
    if need_for_LAI > 0:                      # if yes, then compute the effective LAI for each stand
        import effective_LAI_function
        LAI_dict = effective_LAI_function.effective_LAI( filename_mv, treestandid_inc = treestandid_inc )
        df_LAI = pd.DataFrame()
        df_LAI['standid'] = LAI_dict.keys()
        df_LAI['LAI'] = LAI_dict.values()
        
        # Merge also the computed effective LAI values with the correct standids.
        df_final_dataframe = pd.merge(left=df_final_dataframe, right=df_LAI, left_on='standid', right_on='standid')
    
    
    # =================================================================================================================================================================================
    
    """
    Final DataFrame and function return
    
    """
    # If "maintreespecies" variable was included in the input list, it has been read from two different tables.
    # "maintreespecies" can be found from table 'stand' and 'treestandsummary'. Only in table 'stand' the values are valid, from the other table the values are just Nan.
    # Because the variable is collected from two tables, the dataframe will have two different column names "maintreespecies_x" and "maintreespecies_y".
    # Only "maintreespecies_x" is needed. Hence, it will be renamed and kept in the dataframe that the function will return.
    included_or_not = needed_fields.count('maintreespecies')
    if included_or_not > 0:
        df_final_dataframe = df_final_dataframe.drop('maintreespecies_y', axis=1)
        df_final_dataframe = df_final_dataframe.rename(columns={'maintreespecies_x': 'maintreespecies'})

    return df_final_dataframe