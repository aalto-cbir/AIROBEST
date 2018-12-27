"""
get_holdout_set.py is a function that creates a holdout set for final accuracy evaluation.
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


Inputs:
  - path to csv-file with target and feature values
  - size of the holdout set (in percentages) from the target & feature value dataset
  
Outputs:
  - list of holdout set stand ids

"""

import math
import pandas as pd

def create_holdout_set( csv_file_location, size_of_holdout_set ):
    
    # Read the data
    df = pd.read_csv( csv_file_location, sep=',' )
    stands = df['standid'].values.tolist()
    
    # Shuffle all the indices
    length = int(len(stands)) # the total length
    random_nros = list(range(0, length))
    from random import shuffle # import shuffle 
    shuffle(random_nros) # create list of random indices
    
    # Take some percentage (%) of the data to the holdout set.
    portion = size_of_holdout_set # in percentages (%)
    divisor = math.floor((portion/100) * length)  # compute the partition from the total dataset that corresponds to the portion given above
    
    # Select the first X percent (%) of rows into the holdout set. These rows are in random order now.
    holdout_rows = random_nros[0:divisor]
    
    # Collect the corresponding rows
    stands_in_holdout_set = []
    for indeks in holdout_rows:
        stands_in_holdout_set.append(stands[indeks])
    
    return stands_in_holdout_set