"""
main_function.py is used to call other functions when obtaining Hila data
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
  - Path to Hila geopackage
  - Path to Stand geopackage
  - csv-list of pre-selected stand IDs
  - Raster image
  - Name of the output csv-file

OUTPUT
  - csv-file including Hila grid cell ID and WKT geometries from
    Hila grid cells within the limited area of pre-selected stands


"""

def main( Hila_data, Stand_data, Stand_list, Raster_image, output_name ):
    
    import os
    import time
    currentfolder = os.getcwd()
    starting_time = time.time()
    local_time = time.localtime(time.time())
    hours = local_time[3]
    minutes = local_time[4]
    seconds = local_time[5]
    print("Started at: {:d}:{:d}.{:d}\n".format(hours, minutes, seconds))
    
    
    """
    Section (1)
    Collects the required stand geometries from the list given as function input.
    
    """
    print("Collecting stand geometries...")
    import get_stands
    df_stand = get_stands.getstands(Stand_data, Stand_list)
    print("Done.\n")
    

    """    
    Section (2)
    Collects Hila data within the used raster image.
    
    """
    print("Collecting Hila data...")
    import get_hilacells
    df_hila = get_hilacells.gethilacells(Hila_data, Raster_image)
    print("Done.\n")

    
    """
    Section (3)
    Checks which Hila grid cells are within the stands collected in Section (1).
    
    """
    print("Searching Hila gridcells within the stands...")
    # Stand geometries
    standgeoms = df_stand['geometry'].tolist()
    
    # Hila geometries
    hilageoms = df_hila['geometry'].tolist()
    
    # Hila gridcellids
    gridcellids = df_hila['gridcellid'].tolist()
    
    # Check which Hila grid cells are completely within a stand
    gridcellids_list = []
    for n in range(len(standgeoms)):
        stand_geom = standgeoms[n]
        for k in range(len(hilageoms)):
            Check = stand_geom.Contains(hilageoms[k])
            if Check == True:
                gridcellids_list.append(gridcellids[k])
    
    # Constructing DataFrame
    import pandas as pd
    df_within = pd.DataFrame(data={'gridcellid': gridcellids_list})
    
    # Merging with the earlier one
    df_final_hila = pd.merge(left=df_hila, right=df_within, left_on='gridcellid', right_on='gridcellid')
    print("Done.\n")
    
        
    """
    Section (4)
    Checks which Hila cells are not within the extents of raster data (i.e., NoData for the gridcell).
    
    """
    print("Searching Hila gridcells that have image data...")
    import get_zonalstatistics
    gridcellidlist = df_final_hila['gridcellid'].tolist()
    geometrylist = df_final_hila['geometry'].tolist()
    
    # Call a function to compute zonal majority. (Using only the first band of the used raster image)
    df_stats = get_zonalstatistics.getzonalstatistics(Raster_image, geometrylist)
    
    # Assign gridcellids to the zonal statistics dataframe in order to merge it with the earlier one.
    df_ids = pd.DataFrame(data={'gridcellid': gridcellidlist})
    df_stats = df_stats.assign(gridcellid=df_ids.values)
    
    # Merge the zonal majority values to the earlier dataframe with Hila data and geometries
    df_output = pd.merge(left=df_stats, right=df_final_hila, left_on='gridcellid', right_on='gridcellid')
    
    # Delete all the rows where there is No Data within the geometry, i.e. delete all NaNs.
    df_output = df_output.dropna()
    print("Done.\n")
    
    
    """
    Section (5)
    Write output csv-file
    
    """
    # Write out only two columns: 'gridcellid' and 'geometry'
    df_output = df_output[['geometry', 'gridcellid']]
    
    # Write the output to csv-file that can be opened in QGIS.
    csvfile_name = os.path.join(currentfolder, output_name)
    df_output.to_csv(csvfile_name, encoding='utf-8', index=False)


    m, s = divmod((time.time() - starting_time), 60)
    h, m = divmod(m, 60)
    print("\nAll finished in:")
    print("%dh %02dmin %02dsec " % (h, m, s))