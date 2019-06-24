"""
get_species_specific_data.py function returns species specific tree information.
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


  + Stratum tree species are class values ranging from 1 to 29. 1 == pine and 2 == spruce. We assume that everything else is 3 == broadleaved.
  + OUTPUT ==> Function returns dataframe where:
      - lists of total basal area of species, i.e. total BA of pine, spruce and broadleaved within one stand.
      - In addition to total BA of species, leaf area index (LAI) of different species are returned as well
      - dataframe that will be returned has the following columns:
           'standid', 'pine_BA', 'pine_LB', 'pine_LAI', 'spruce_BA', 'spruce_LB', 'spruce_LAI', 'broadleaved_BA', 'broadleaved_LB', 'broadleaved_LAI'

INPUTS:
  + geopackage_name (e.g. "MV_Ruovesi.gpkg")
  + list of stand IDs
  + treestandid_inc: increment to get treestandid from standid
      - Different values can be used to get the different scenarios -- 1000000000, 2000000000 or 3000000000
      - 2000000000 should get the recent values, 3000000000 is future projection, and 1000000000 are the measurements
      - Corresponds to the types found in the geopackage files (type=1, type=2 and type=3)

"""

import numpy as np
import pandas as pd
from hypdatatools_img import *
from hypdatatools_gdal import *
import stratum_effective_LAI_function

def getstandtreespeciesdata(standid_list, treestandid_inc, gpkg_path):
    geopackage_location = gpkg_path
    
    # Calculate treestandids from standids
    treestandid_list = [ i + treestandid_inc for i in standid_list ]
    
    # Combine standid and treestandid into one single Dataframe
    d_standid = {'standid':standid_list, 'treestandid':treestandid_list}  
    df_standid = pd.DataFrame(data=d_standid)
    
    
    # =================================================================================================================================================================================
    
    """
    Fields from table "treestratum"
    
    """
    conn = sqlite3.connect( geopackage_location )
    c = conn.cursor()
    
    # Get the values from the fields we need
    table_id_treestratum_list    = geopackage_getvalues( c, "treestratum", "id",          additionalconstraint=None )
    treestandid_treestratum_list = geopackage_getvalues( c, "treestratum", "treestandid", additionalconstraint=None )
    treespecies_treestratum_list = geopackage_getvalues( c, "treestratum", "treespecies", additionalconstraint=None )
    basalarea_treestratum_list   = geopackage_getvalues( c, "treestratum", "basalarea",   additionalconstraint=None )
    conn.close()
    
    # New order for 'treestandid' --> this because the treestandids are not in the right order. The correct order is told in the id list of current table.
    new_treestandid_list = [np.nan] * (len(treestandid_treestratum_list))
    j = 0
    for indexvalue in table_id_treestratum_list:
        new_treestandid_list[indexvalue-1] = treestandid_treestratum_list[j]
        j += 1
     
    # Constructing DataFrame
    d_dt = {'treestandid': new_treestandid_list, 'stratum_BA': basalarea_treestratum_list, 'stratum_tree': treespecies_treestratum_list}
    df_treestratum_table = pd.DataFrame(data=d_dt)
    
    # Merge with the earlier dataframe (using: 'treestandid')
    df_dataframe = pd.merge(left=df_standid, right=df_treestratum_table, left_on='treestandid', right_on='treestandid')
    
    # Column 'stratum_tree' has all the stratum tree species. Let's modify this column so that there will be just three classes 1-pine, 2-spruce, 3-broadleaved.
    # Let's convert all class numbers to 3, apart from classes 1 and 2.
    stratum_tree_column_as_list = df_dataframe['stratum_tree'].tolist()
    all_species = list(set(stratum_tree_column_as_list)) # all different tree species we have in the dataset
    all_species_dropped_pine = list(filter(lambda a: a != 1, all_species)) # drop pine classes away
    all_species_dropped_pine_and_spruce = list(filter(lambda a: a != 2, all_species_dropped_pine)) # drop also spruce classes away
    
    for j_species in all_species_dropped_pine_and_spruce:
        item_to_replace = j_species
        replacement_value = 3
        indices_to_replace = [i for i,x in enumerate(stratum_tree_column_as_list) if x==item_to_replace]
        for i in indices_to_replace:
            stratum_tree_column_as_list[i] = replacement_value
    
    # Check all different tree species that we have at this point. Should be only three species: Pine-1  Spruce-2  Broadleaved-3.
    all_tree_species = list(set(stratum_tree_column_as_list))
    if len(all_tree_species) > 3:
        import sys
        sys.exit("Error: More than 3 different tree species")
    
    # Replace the 'stratum_tree' column in the dataframe with the new class list
    df_dataframe = df_dataframe.drop('stratum_tree', axis=1) # first drop the old column
    d_trees  = {'stratum_tree': stratum_tree_column_as_list}
    df_st_trees = pd.DataFrame(data=d_trees)
    df_dataframe = df_dataframe.assign(stratum_tree=df_st_trees.values) # add the new list as column
    
    # Rearrange the columns
    cols = [ 'standid', 'treestandid', 'stratum_BA', 'stratum_tree']
    df_dataframe = df_dataframe[cols]
    
    
    # =================================================================================================================================================================================
    
    """
    (2) Collect tree species specific information within one stand:
        - basal area (BA)
        - Leaf area index (LAI)
    
    This means that e.g. all strata that have pine as their main tree species, are gathered and their basal areas are added up.
    Hence, a variable called pine_BA is created, and it describes the amount of basal area of pine trees within one single stand.
    
    """
    
    # ==============   (1) LEAF AREA INDEX   ==============
    
    # We start by collecting effective LAI for each stand strata.
    
    # Output: two lists where the first element is the correct standID, and rest of the list elements are either strata tree species or strata LAI values
    list_species, list_LAI_values = stratum_effective_LAI_function.stratum_effective_LAI(geopackage_location, treestandid_inc)
    
    # Gather standIDs, LAI values and tree species in their own lists
    all_stands = [] # stands are the same in both lists 
    strata_LAI = []
    strata_species = []
    for j in range(len(list_species)):
        all_stands.append(list_species[j][0])
        strata_LAI.append(list_LAI_values[j][1:])
        strata_species.append(list_species[j][1:])
    
    # Now check which are the stands we actually are interested in (stands at the moment).
    stands_atm = df_dataframe['standid'].tolist()
    
    # Intersection with the stands of interest and each standid from the function that computed effective LAI of strata
    intersektion = np.intersect1d(all_stands, stands_atm, assume_unique=False, return_indices=True)  # output has the indices at index [1]
    the_indices = (list(intersektion[1]))
    index_list = sorted(the_indices) # sort the indices starting from the smallest index
        
    # Collect data only from the wanted indices / stands
    stand_lst = []
    strata_tree_species_LAI = []
    strata_LAI_values = []
    for index in index_list:
        stand_lst.append(all_stands[index])
        strata_tree_species_LAI.append(strata_species[index])
        strata_LAI_values.append(strata_LAI[index])
    
    # We need to again convert all class numbers to 3, apart from classes 1 and 2.
    for j in range(len(strata_tree_species_LAI)):
        list_of_different_tree_species = list(set(strata_tree_species_LAI[j]))
        different_species_dropped_pine = list(filter(lambda a: a != 1, list_of_different_tree_species)) # drop pine classes away
        different_species_dropped_pine_and_spruce = list(filter(lambda a: a != 2, different_species_dropped_pine))  # drop also spruce classes away
        
        for j_species in different_species_dropped_pine_and_spruce:
            item_to_replace = j_species
            replacement_value = 3
            indices_to_replace = [i for i,x in enumerate(strata_tree_species_LAI[j]) if x==item_to_replace]
            for i in indices_to_replace:
                strata_tree_species_LAI[j][i] = replacement_value
        
        # Check all different tree species that we have at this point. Should be only three species: Pine-1  Spruce-2  Broadleaved-3.
        all_tree_species_atm = list(set(strata_tree_species_LAI[j]))
        if len(all_tree_species_atm) > 3:
            import sys
            sys.exit("Error: More than 3 different tree species")
    
    
    # ==============   (2) BASAL AREA    ==============
    
    # Next we collect basal area for each strata.
    # In addition, we create lists for each tree species and variable. E.g. list for pine LAI strata -- pine_LAI
    
    """
    Needed dataframe columns as lists
    
    """
    
    # Stands
    stand_list = df_dataframe['standid'].tolist()
      
    # Strata  
    stratum_basalarea_column_as_list = df_dataframe['stratum_BA'].tolist()
    stratum_treespecies_column_as_list = df_dataframe['stratum_tree'].tolist()
    
    # Empty lists for storing results
    pine_BA_list = []
    spruce_BA_list = []
    broadleaved_BA_list = []
    #
    pine_LAI_list = []
    spruce_LAI_list = []
    broadleaved_LAI_list = []
    #
    current_stand_list = []
    
    m=0
    j = 0
    for g in range(len(stand_list)):
        stand_ID = stand_list[j]
        
        # Find the indices where we have the same stand
        indices = [ i for i, x in enumerate(stand_list) if x == stand_ID]
        
        # At this point we have the desired indices per one stand
        # Collect every corresponding:
        #  - stratum basal areas
        #  - stratum tree species
        st_BA   = [ stratum_basalarea_column_as_list[i] for i in indices ]
        st_tree = [ stratum_treespecies_column_as_list[i] for i in indices ]
        
        # Stratum LAI values are already in list called 'strata_LAI_values'
        st_LAI = strata_LAI_values[m]
        
        
        # One stand has several strata. One stratum has one main tree species. Let's read all different species in this one particular stand,
        # and collect all the basal areas (or LAIs) of this particular tree species within one stand.
        
        # Every stratum treespecies in one stand
        Tree_Species_in_Stand = list(set(st_tree))
        
        total_species_BA   = []
        total_species_LAI  = []
        for Species in Tree_Species_in_Stand:
            # Index where data of this particular tree species locates in this stand
            indeks = [ i for i, x in enumerate(st_tree) if x == Species ]
            
            # With indeks we get BA, LB and LAI for this particular species, and with for-loop we get the total within this entire stand.
            gather_species_BA  = []
            gather_species_LAI = []
            for k in indeks:
                BA_for_this_indeks  = st_BA[k]
                LAI_for_this_indeks = st_LAI[k]
                gather_species_BA.append(BA_for_this_indeks)
                gather_species_LAI.append(LAI_for_this_indeks)
    
            total_species_BA.append(sum(gather_species_BA))
            total_species_LAI.append(sum(gather_species_LAI))
            
        # At this point we have the total basal areas, and leaf area indices (LAIs) of tree species per one stand in a list.
        # Next we will collect these basal areas, and leaf area indices to species specific lists.
                
        # Pine data
        pine_index = [i for i,v in enumerate(Tree_Species_in_Stand) if v==1] # search which index has 1-pine as class value in list 'Tree_Species_in_Stand'
        if len(pine_index) == 0:
            # This stand has no pine strata
            pine_BA_list.append(0)
            pine_LAI_list.append(0)
        else:
            pine_BA_list.append(total_species_BA[pine_index[0]])
            pine_LAI_list.append(total_species_LAI[pine_index[0]])
            
        # Spruce data
        spruce_index = [i for i,v in enumerate(Tree_Species_in_Stand) if v==2] 
        if len(spruce_index) == 0:
            # This stand has no spruce strata
            spruce_BA_list.append(0)
            spruce_LAI_list.append(0)
        else:
            spruce_BA_list.append(total_species_BA[spruce_index[0]])
            spruce_LAI_list.append(total_species_LAI[spruce_index[0]])
            
        # Broadleaved data
        broadleaved_index = [i for i,v in enumerate(Tree_Species_in_Stand) if v==3] 
        if len(broadleaved_index) == 0:
            # This stand has no broadleaved strata
            broadleaved_BA_list.append(0)
            broadleaved_LAI_list.append(0)
        else:
            broadleaved_BA_list.append(total_species_BA[broadleaved_index[0]])
            broadleaved_LAI_list.append(total_species_LAI[broadleaved_index[0]])
        
        current_stand_list.append(stand_ID)
        
        m += 1
        
        # If the next index value gets too big, i.e. exceeds the list index, we will stop the loop.
        if (j + len(indices)) > (len(stand_list)-1):
            break
        else:
            j = j + len(indices)
        
        # End of for-loop
        
    # Create dataframe where we have standid, pine_BA, pine_LAI, spruce_BA, spruce_LAI, broadleaved_BA and broadleaved_LAI
    d_tree_info = {'standid': current_stand_list, 'pine_BA': pine_BA_list, 'pine_LAI': pine_LAI_list , 'spruce_BA': spruce_BA_list,
                   'spruce_LAI': spruce_LAI_list, 'broadleaved_BA': broadleaved_BA_list, 'broadleaved_LAI': broadleaved_LAI_list }
    df_tree_information = pd.DataFrame(data=d_tree_info)
    
    # Rearrange the columns
    Columns = ['standid', 'pine_BA', 'pine_LAI', 'spruce_BA', 'spruce_LAI', 'broadleaved_BA', 'broadleaved_LAI']
    df_tree_information = df_tree_information[Columns]
    
    
    """
    Function return
    
    """
    # End of function. Return the final dataframe.
    return df_tree_information