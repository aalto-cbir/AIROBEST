"""
get_tree_species.py is a function that computes main tree species of a stand based on basal area (BA) or leaf area index (LAI)
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


Input:
  - dataframe full of species specific information obtained from function get_species_specific_data.py

Example: df_main_tree_species = get_tree_species.getmaintreespecies( df_species_specific_data )

"""

import numpy as np
import pandas as pd

def getmaintreespecies( df_species_specific_data ):
    
    df = df_species_specific_data
    standid_lst = df['standid'].tolist()
    
    # --- Basal area ---
    # add up
    true_stand_BA = np.asarray(df['pine_BA'].values.tolist()) + np.asarray(df['spruce_BA'].values.tolist()) + np.asarray(df['broadleaved_BA'].values.tolist())
    # assign to the dataframe
    d_true_BA  = {'stand_BA': true_stand_BA.tolist()}
    df_BA = pd.DataFrame(data=d_true_BA)
    df = df.assign(stand_BA=df_BA.values)

    # --- Leaf area index ---
    # add up
    true_stand_LAI = np.asarray(df['pine_LAI'].values.tolist()) + np.asarray(df['spruce_LAI'].values.tolist()) + np.asarray(df['broadleaved_LAI'].values.tolist())
    # assign to the dataframe
    d_true_LAI  = {'stand_LAI': true_stand_LAI.tolist()}
    df_LAI = pd.DataFrame(data=d_true_LAI)
    df = df.assign(stand_LAI=df_LAI.values)
    
    # Percentages of true values.
    true_pine_BA_perc = 100 * np.asarray(df['pine_BA'].values.tolist()) / true_stand_BA
    true_spruce_BA_perc = 100 * np.asarray(df['spruce_BA'].values.tolist()) / true_stand_BA
    true_broadleaved_BA_perc = 100 * np.asarray(df['broadleaved_BA'].values.tolist()) / true_stand_BA
    
    true_pine_LAI_perc = 100 * np.asarray(df['pine_LAI'].values.tolist()) / true_stand_LAI
    true_spruce_LAI_perc = 100 * np.asarray(df['spruce_LAI'].values.tolist()) / true_stand_LAI
    true_broadleaved_LAI_perc = 100 * np.asarray(df['broadleaved_LAI'].values.tolist()) / true_stand_LAI
    
    # If the maximum percentage is less than 75 %, assign -99999 as class value.
    true_BA_main_tree_classes = []
    for i in range(len(df)):
        percentages = [ true_pine_BA_perc[i], true_spruce_BA_perc[i], true_broadleaved_BA_perc[i] ]
        max_index = [ i for i,v in enumerate(percentages) if v==(max(percentages)) ]
        if max_index[0] == 0:
            if max(percentages)<75:
                true_BA_main_tree_classes.append(-99999)
            else:
                true_BA_main_tree_classes.append(1)
        if max_index[0] == 1:
            if max(percentages)<75:
                true_BA_main_tree_classes.append(-99999)
            else:
                true_BA_main_tree_classes.append(2)
        if max_index[0] == 2:
            if max(percentages)<75:
                true_BA_main_tree_classes.append(-99999)
            else:
                true_BA_main_tree_classes.append(3)
    
    true_LAI_main_tree_classes = []
    for i in range(len(df)):
        percentages = [ true_pine_LAI_perc[i], true_spruce_LAI_perc[i], true_broadleaved_LAI_perc[i] ]
        max_index = [ i for i,v in enumerate(percentages) if v==(max(percentages)) ]
        if max_index[0] == 0:
            if max(percentages)<75:
                true_LAI_main_tree_classes.append(-99999)
            else:
                true_LAI_main_tree_classes.append(1)
        if max_index[0] == 1:
            if max(percentages)<75:
                true_LAI_main_tree_classes.append(-99999)
            else:
                true_LAI_main_tree_classes.append(2)
        if max_index[0] == 2:
            if max(percentages)<75:
                true_LAI_main_tree_classes.append(-99999)
            else:
                true_LAI_main_tree_classes.append(3) 
    
    # Write these to the dataframes
    d_perc_true = {'species_BA': true_BA_main_tree_classes, 'species_LAI': true_LAI_main_tree_classes, 'standid': standid_lst}
    df_perc_true = pd.DataFrame(data=d_perc_true)
    df = pd.merge(left=df, right=df_perc_true, left_on='standid', right_on='standid')
    
    # Choose only the columns you will need
    cols = ['standid', 'species_BA', 'species_LAI', 'pine_BA', 'pine_LAI', 'spruce_BA', 'spruce_LAI', 'broadleaved_BA', 'broadleaved_LAI']
    df = df[cols]

    return df