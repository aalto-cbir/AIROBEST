"""
Copyright (C) 2017,2018  Matti Mõttus 
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Extracts standwise data vectors from Finnish Forestry Center standwise data
   merge data from different tables (stand, treestand, treestandsummary, treestratum)
   to create a data vector for each plot. See code for specific format of the data vector
"""
import numpy as np
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import time
import os
import datetime

# import the hyperspectral libraries and other things from tools
# Pyzo needs these commands to be run from a file in this folder, otherwise it cannot find tools (??)
import sys
import importlib
sys.path.append("C:\\Users\\MMATTIM\\OneDrive - Teknologian Tutkimuskeskus VTT\\koodid\\python\\hyperspectral\\AIROBEST")
from tools.hypdatatools_gdal import * 


filenames_mv = [ "D:/mmattim/wrk/hyytiala-D/mv/MV_Juupajoki.gpkg", "D:/mmattim/wrk/hyytiala-D/mv/MV_Ruovesi.gpkg" ]

datafolder = 'D:/mmattim/wrk/hyytiala-D/AISA2017'
# hyperspectral_filename = '20170615_reflectance_mosaic_128b.hdr' 

outfolder = 'D:/mmattim/wrk/hyytiala-D/temp'
filename_out = os.path.join( outfolder, "forestdata.csv")

# -----------------------------------------------------
# read metsäkeskus geopackages
# rasterize the selected data layers based on an already existing raster
#     existing raster is only used for defining the output geometry
# see code for details on which layers are rasterized
#   (the rasterized variables names area also printed out)

fieldnames_in = [ 'standid', 'fertilityclass', 'soiltype' ] 
# maintreespecies in treestandsummary is not trustworthy -- it is often empty


outlist = [ [], [] ] + [ [] for i in fieldnames_in ] # geometries and other data from the standid table
outlist_extra = [] # data combined from other data tables (list of lists)
outlist_extranames = [] # names of the variables in outlist_extra

for filename_mv in filenames_mv:
    outlist_i = ( vector_getfeatures( filename_mv, fieldnames_in ) )
    # outlist_i and outlist contain n+2 sublists: FID, geometries, standid, fertilityclass, etc (see fieldnames_in)

    # copy to the big outlist
    for l,i in zip(outlist,outlist_i):
        l += i
    standids_i = outlist_i[2]

    # open the data table and keep it open for a while
    conn = sqlite3.connect( filename_mv )
    c = conn.cursor()

    listcounter = 0  # count how many elements are already in outlist_extra

    # -------- Area (geometric, in projection data units)
    standarea = [ i.GetArea() for i in outlist_i[1] ]
    if len(outlist_extra) < listcounter + 1:
        outlist_extra.append( standarea )
        outlist_extranames.append( "stand_area" )
    else:
        outlist_extra[listcounter] += standarea
    listcounter += 1
                
    # -------- stemcount from treestandsummary [integer]
    # generate treestandid from standid. For the data modeled to beginning of 2018,  it should be 2000000000+standid
    tsids_i = [ i + 2000000000 for i in standids_i ]
    q=geopackage_getspecificvalues1(c,"treestandsummary","stemcount",tsids_i,"treestandid")
    stemcount = [ i[0][0] if len(i)>0 else 0 for i in q ]
    if len(outlist_extra) < listcounter + 1:
        outlist_extra.append( stemcount )
        outlist_extranames.append( "stemcount" )
    else:
        outlist_extra[listcounter] += stemcount
    listcounter += 1    
    # -------- meanheight from treestandsummary [float] 
    q=geopackage_getspecificvalues1(c,"treestandsummary","meanheight",tsids_i,"treestandid")
    meanheight = [ i[0][0] if len(i)>0 else 0 for i in q ]
    if len(outlist_extra) < listcounter + 1:
        outlist_extra.append( meanheight )
        outlist_extranames.append( "meanheight" )
    else:
        outlist_extra[listcounter] += meanheight
    listcounter += 1    
    # Get the number of tree strata from treestratum [integer], retrieve the basal area for each
    q_basal=geopackage_getspecificvalues1(c,"treestratum","basalarea",tsids_i,"treestandid")
    # q_basal will also be used later
    N_strata = [ len(i) for i in q_basal ]
    
    
    # -------- main species, and percentage of mean species in basalarea from treestratum [integer]
    #                            percentages of pine, spruce and broadleaf [integer]
    #
    q_species=geopackage_getspecificvalues1(c,"treestratum","treespecies",tsids_i,"treestandid")
    # q_species will also be used later
    percentage_mainspecies = []
    maintreespecies = []
    percentage_pine = []
    percentage_spruce = []
    percentage_broadleaf = []
    for qi, q1i in zip( q_basal, q_species ):
        if len(qi) > 0:
            basalareas = [ i[0] for i in qi ]
            if sum( basalareas ) > 0:
                species = [ i[0] for i in q1i ]
                # first, find main species. find all species present and find the basal area for each
                uniquespecies = list( set(species) )
                basalarea_uniquespecies = [ sum( [ ba_k for ba_k,sp_k in zip(basalareas,species) if sp_k==sp_j ] ) for sp_j in uniquespecies ]
                i_main = basalarea_uniquespecies.index( max(basalarea_uniquespecies) )
                basalarea_mainspecies = basalarea_uniquespecies[ i_main ] 
                maintreespecies.append( uniquespecies[ i_main ] )
                sumbasalareas = sum(basalareas)
                percentage_mainspecies.append( int( basalarea_mainspecies / sumbasalareas * 100 ) )
                # percentages by species
                if 1 in uniquespecies:
                    percentage_pine.append( int( basalarea_uniquespecies[ uniquespecies.index( 1 ) ] / sumbasalareas * 100 ) )
                else:
                    percentage_pine.append( 0 )
                if 2 in uniquespecies:
                    percentage_spruce.append( int( basalarea_uniquespecies[ uniquespecies.index( 2 ) ] / sumbasalareas * 100 ) )
                else:
                    percentage_spruce.append( 0 )
                # merge all others into broadleaves
                # a computationally more efficient way would be to to assume that all the rest is broadleaf
                i_broadleaf = [ i for i in range(len(uniquespecies)) if uniquespecies[i]>2 ]
                if len( i_broadleaf ) > 0:
                    percentage_broadleaf.append( int( sum( [ basalarea_uniquespecies[j] for j in i_broadleaf ] ) / sumbasalareas * 100 ) )
                else:
                    percentage_broadleaf.append( 0 )
            else:
                maintreespecies.append( 0 )
                percentage_mainspecies.append( 0 )
                percentage_pine.append( 0 )
                percentage_spruce.append( 0 )
                percentage_broadleaf.append( 0 )
        else:
            maintreespecies.append( 0 )
            percentage_mainspecies.append( 0 )
            percentage_pine.append( 0 )
            percentage_spruce.append( 0 )
            percentage_broadleaf.append( 0 )
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( maintreespecies )
        outlist_extranames.append( "maintreespecies" )
        outlist_extra.append( percentage_mainspecies )
        outlist_extranames.append( "percentage_mainspecies" )
        outlist_extra.append( percentage_pine)
        outlist_extranames.append( "percentage_pine" )
        outlist_extra.append( percentage_spruce )
        outlist_extranames.append( "percentage_spruce" )
        outlist_extra.append( percentage_broadleaf )
        outlist_extranames.append( "percentage_broadleaf" )
    else:
        outlist_extra[listcounter] += maintreespecies
        outlist_extra[listcounter+1] += percentage_mainspecies
        outlist_extra[listcounter+2] += percentage_pine
        outlist_extra[listcounter+3] += percentage_spruce
        outlist_extra[listcounter+4] += percentage_broadleaf
    listcounter += 5
    
    # -------- dbh from treestandsummary [float]
    q=geopackage_getspecificvalues1(c,"treestandsummary","meandiameter",tsids_i,"treestandid")
    dbh = [ i[0][0] if len(i)>0 else 0 for i in q ]
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( dbh )
        outlist_extranames.append( "dbh" )
    else:
        outlist_extra[listcounter] += dbh
    listcounter += 1
    
    
    #                      three main strata: species, height, dbh, density
    #
    q_dbh=geopackage_getspecificvalues1(c,"treestratum","meandiameter",tsids_i,"treestandid")
    q_height=geopackage_getspecificvalues1(c,"treestratum","meanheight",tsids_i,"treestandid")
    # for each plot, reorder strate according to basal area
    species_1 = [] # the lists from the most important strata
    dbh_1 = []
    height_1 = []
    density_1 = []
    species_2 = []
    dbh_2 = []
    height_2 = []
    density_2 = []
    species_3 = []
    dbh_3 = []
    height_3 = []
    density_3 = []
    for qb,qs,qd,qh in zip( q_basal, q_species, q_dbh, q_height ):
        if len(qb) > 0:
            i_sort = np.flip( np.argsort( [ i[0] for i in qb] ),0 )
                # qb needs to be unpacked, and then the resulting index flipped
            species_1.append( qs[ int(i_sort[0]) ][0] ) 
            dbh_1.append( qd[ int(i_sort[0]) ][0] ) 
            height_1.append( qh[ int(i_sort[0]) ][0] )  
            diam = qd[ int(i_sort[0]) ][0]**2*np.pi*1e-4  # diameter was in cms
            if diam > 0:
                density_1.append( qb[ int(i_sort[0]) ][0]/diam )
            else:
                density_1.append( 0. )
                
            if len(qb)>1:
                species_2.append( qs[ int(i_sort[1]) ][0] ) 
                dbh_2.append( qd[ int(i_sort[1]) ][0] ) 
                height_2.append( qh[ int(i_sort[1]) ][0] )  
                diam = qd[ int(i_sort[1]) ][0]**2*np.pi*1e-4  # diameter was in cms
                if diam > 0:
                    density_2.append( qb[ int(i_sort[1]) ][0]/diam )
                else:
                    density_2.append( 0. )
                if len(qb)>2:
                    # we have at least three strata
                    species_3.append( qs[ int(i_sort[2]) ][0] ) 
                    dbh_3.append( qd[ int(i_sort[2]) ][0] ) 
                    height_3.append( qh[ int(i_sort[2]) ][0] )  
                    diam = qd[ int(i_sort[2]) ][0]**2*np.pi*1e-4  # diameter was in cms
                    if diam > 0:
                        density_3.append( qb[ int(i_sort[2]) ][0]/diam )
                    else:
                        density_3.append( 0. )
                else:
                    # two strata for stand
                    species_3.append( 0 )
                    dbh_3.append( 0. )
                    height_3.append( 0. )
                    density_3.append( 0. )                
            else: # just one stratum
                species_2.append( 0 )
                dbh_2.append( 0. )
                height_2.append( 0. )
                density_2.append( 0. )            
                species_3.append( 0 )
                dbh_3.append( 0. )
                height_3.append( 0. )
                density_3.append( 0. )
        else: # no strata for this stand
            species_1.append( 0 )
            dbh_1.append( 0. )
            height_1.append( 0. )
            density_1.append( 0. )            
            species_2.append( 0 )
            dbh_2.append( 0. )
            height_2.append( 0. )
            density_2.append( 0. )            
            species_3.append( 0 )
            dbh_3.append( 0. )
            height_3.append( 0. )
            density_3.append( 0. )
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( species_1 )
        outlist_extranames.append( "treespecies_stratum1" )
        outlist_extra.append( dbh_1 )
        outlist_extranames.append( "dbh_stratum1" )
        outlist_extra.append( height_1 )
        outlist_extranames.append( "treeheight_stratum1" )
        outlist_extra.append( density_1 )
        outlist_extranames.append( "trees/ha_stratum1" )
        
        outlist_extra.append( species_2 )
        outlist_extranames.append( "treespecies_stratum2" )
        outlist_extra.append( dbh_2 )
        outlist_extranames.append( "dbh_stratum2" )
        outlist_extra.append( height_2 )
        outlist_extranames.append( "treeheight_stratum2" )
        outlist_extra.append( density_2 )
        outlist_extranames.append( "trees/ha_stratum2" )
        
        outlist_extra.append( species_3 )
        outlist_extranames.append( "treespecies_stratum3" )
        outlist_extra.append( dbh_3 )
        outlist_extranames.append( "dbh_stratum3" )
        outlist_extra.append( height_3 )
        outlist_extranames.append( "treeheight_stratum3" )
        outlist_extra.append( density_3 )
        outlist_extranames.append( "trees/ha_stratum3" )

    else:
        outlist_extra[listcounter] += species_1
        outlist_extra[listcounter+1] += dbh_1
        outlist_extra[listcounter+2] += height_1
        outlist_extra[listcounter+3] += density_1
        
        outlist_extra[listcounter+4] += species_2
        outlist_extra[listcounter+5] += dbh_2
        outlist_extra[listcounter+6] += height_2
        outlist_extra[listcounter+7] += density_2
        
        outlist_extra[listcounter+8] += species_3
        outlist_extra[listcounter+9] += dbh_3
        outlist_extra[listcounter+10] += height_3
        outlist_extra[listcounter+11] += density_3
    listcounter += 12

    conn.close()

# exclude first two columns (FID, geometries) from saving as csv
# and select the relevant columns from outlist_extra
i_extra = np.array([0,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
outfeatures = outlist[ 2: ] +[ outlist_extra[i] for i in range(len(outlist_extra)) if i in i_extra]
outnames =  fieldnames_in + [ outlist_extranames[i] for i in range(len(outlist_extranames)) if i in i_extra]

headerline = "\t".join(outnames)
='\t', header=headerline )
