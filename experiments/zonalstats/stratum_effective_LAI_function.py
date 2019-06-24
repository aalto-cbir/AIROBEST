"""
stratum_effective_LAI_function.py computes effective leaf area index (LAI) from the data of Metsakeskus for each stratum within the stands.
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


 About this function:
     
   + file_in: filename of the geopackage file to be used
   + treestandid_inc: increment to get treestandid from stadid
       - Different values can be used to ge the different scenarios -- 1000000000, 2000000000 or 3000000000
       - 2000000000 should get the recent values, 3000000000 is future projection
       - Corresponds to the types found in the geopackage files (type=1, type=2 and type=3)
       
   Function returns: 2 lists. First element of lists is the standid and rest of the elements are strata values (tree species or LAI values).
    
   Processing is done in the following steps:
       * read standids from the geopackage
       * convert standids to treestandids, which are used in "treestratum" table
       * retrieve all strata for each (tree)standid
           - extract species and leaf biomass for each stratum
           - convert leaf biomass to 'true' leaf area ( Assume leaf biomass as: tons/ha = 1000kg/10000m2 = 1kg/10m2 = 1000g/10m2 = 100g/m2 = 100*(g/m2) )
           - weight 'true' leaf area with species specified STAR-value in order to get 'effective' leaf area

"""

import borealallometry
from hypdatatools_gdal import *

def stratum_effective_LAI(file_in, treestandid_inc):

    conn = sqlite3.connect( file_in )
    c = conn.cursor()
    
    # Get the standid values
    outlist = vector_getfeatures( file_in, ["standid"] )
    
    # Outlist contains 3 sublists: FID, geometries, standid
    standids = outlist[2]
    
    # Calculate treestandids from standids
    treestandids = [ i + treestandid_inc for i in standids ]
    
    
    """
       Sum over all strata and use specific leaf weight for each species to get LAI.
       Two fields are relevant to LAI calculations: species and leaf biomass.
       
    """
    # Get treespecies and leaf biomass from table named "treestratum", but only from the rows
    # where field named "treestandid" equals to the values in list 'treestandids'
    # Output: List where one element includes as many values as there exists treestrata in the corresponding treestand
    gpkg_species     = geopackage_getspecificvalues1( c, "treestratum", "treespecies", treestandids, "treestandid" )
    gpkg_leafbiomass = geopackage_getspecificvalues1( c, "treestratum", "leafbiomass", treestandids, "treestandid" )
    
    conn.close()
    
    # Create empty lists for outputs
    stand_and_species = []
    stand_and_LAI_values = []
    
    
    # Go through all standids
    for species_i, leafbiomass_i, stand_i in zip( gpkg_species, gpkg_leafbiomass, standids ):   # Goes through each row of the lists in parentheses.
        
        # Check whether this particular standid had any tree stratums inside it.
        if len(species_i) > 0:
            
            # Outputs: lists of leafbiomasses and tree species within one particular treestand (may include several strata, i.e. several values in list)
            leafbiomass = [ i[0] for i in leafbiomass_i ]  # we use i[0] because the values are like this: [(2,), (1,), (1,)]
            species     = [ i[0] for i in species_i ]


            """
            Append current stand ID to be the first element, append strata tree species list after that.
            """
            stand_and_species.append([stand_i]+species)
            
            
            # Leafbiomass (above) includes leaf biomasses of each stratum from one specific treestand.
            # Sum of this list tells the leaf biomass of one specific treestand.
            stand_biomass = sum(leafbiomass)
            
            # If there's some leaf biomass in the treestand, we start computing effective leaf area index (LAI)
            if stand_biomass > 0:
                
                # s_leaf_weight: Specific leaf weight (g/m2). Value differs between tree species.
                # STAR: Spherically averaged shoot silhouette to total area ratio. Value differs between tree species.
                # Outputs: Lists of specific leaf weights and species-specific STAR. Length of the lists depends on how many strata there is in this particular treestand.
                s_leaf_weight = [ borealallometry.slw(i)  for i in species ]
                STAR          = [ borealallometry.STAR(i) for i in species ]
                
                # Output: List including element-wise operation between two lists; treestand leafbiomass and treestand specific leaf weight. Strata/elements are weighted with correct STAR.
                # Units: leafbiomass; 100*(g/m2)  &  specific leaf weight; 1*(g/m2)  &  STAR; unitless
                stratum_operation = [ ((stratum_leafbiomass*100) / stratum_s_leaf_weight)*(4*stratum_STAR) for stratum_leafbiomass, stratum_s_leaf_weight, stratum_STAR in zip(leafbiomass, s_leaf_weight, STAR) ]
                
                
                """
                Append current stand ID to be the first element, append strata effective LAI values list after that.
                """
                stand_and_LAI_values.append([stand_i]+stratum_operation)
                
                
            else:
                stand_and_LAI_values.append([stand_i]+([0]*len(species_i)))
                
        else:
            stand_and_species.append([stand_i]+[0])
            stand_and_LAI_values.append([stand_i]+[0])
    
    return stand_and_species, stand_and_LAI_values