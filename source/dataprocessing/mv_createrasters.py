"""
Copyright (C) 2018,2019,2020  Matti Mõttus 
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Rasterizes Finnish Forestry Center standwise data to create TAIGA reference raster dataset.

"""
import numpy as np
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import time
import os
import datetime
import sqlite3
import json
import scipy.ndimage

# load the hyperspectral functions -- not yet a module
#   -- note: these should be soon available as part of spectralinvariants python module (on GitHub)
#   add the folder with these functions to python path
import sys
# this script is in source/dataprocessing subfolder, include project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import tools.hypdatatools_img
import tools.hypdatatools_gdal
import tools.borealallometry

# input files: gpkg files with FFC data, hyperspectral image
TAIGAdatafolder = 'C:/data/wrk/TAIGA'
hyperspectral_filename = '20170615_reflectance_128b_mosaic.hdr' 
# these files are not part of TAIGA dataset, they are downloaded from https://metsaan.fi
filenames_FFC = ['MV_Juupajoki.gpkg', 'MV_Ruovesi.gpkg' ]
datafolder_FFC = 'C:/data/wrk/TAIGA_source'
# a file with a list of "bad" stand ids, these are known to be incorrect in the database
badstandfile = "bad_stands_updated_15092020.csv"

# output
outfolder = TAIGAdatafolder
filename_newraster = os.path.join( outfolder, 'forestdata_stands')

# Fill in species-specific STAR and slw
# vectors over species: 1=pine, 2=spruce, 3=birch
# STAR[0],slw[0] will remain None
STAR = [None]*4
slw = [None]*4
for i in range( 1, 4 ):
    STAR[i] = tools.borealallometry.STAR( i )
    slw[i] = tools.borealallometry.slw( i )

# read bad stand list: one id per line
f=open( os.path.join( datafolder_FFC, badstandfile ), "r" )
badstandlist = f.read().split('\n')
# retain only the numeric (integer) values, assume others comments
badstandlist = [ int(i) for i in badstandlist if i.strip().isdigit() ]

# -----------------------------------------------------
# read FFC geopackages
# rasterize the selected data layers based on an already existing raster
#     existing raster is only used for defining the output geometry
# see code for details on which layers are rasterized
#   (the rasterized variables names area also printed out)

# which fields can be read directly from the files
fieldnames_in = [ 'standid', 'fertilityclass', 'soiltype' ] 

outlist = [ [], [] ] + [ [] for i in fieldnames_in ] # geometries and other data from the standid table
#   reserve two front locations in outlist for FID and geometry
# Not all data can be read directly. These will be added to outlist_extra later, and merged before saving  
outlist_extra = [] # data combined from other data tables (list of lists)
outlist_extranames = [] # names of the variables in outlist_extra

filename_AISA = os.path.join(TAIGAdatafolder,hyperspectral_filename)
# Loop over the FFC data files and collect required data into outlist_extra
# NOTE: data will be saved in outlist_extra in the order they are processed below
for fi in filenames_FFC:
    filename_mv = os.path.join( datafolder_FFC, fi ) 
    outlist_i = ( tools.hypdatatools_gdal.vector_getfeatures( filename_mv, fieldnames_in ) )
    # outlist contains n+2 sublists: FID, geometries + the fields specified in fieldnames_in
    geomlist_i = outlist_i[1]
    
    i_stand = tools.hypdatatools_gdal.geometries_subsetbyraster( geomlist_i, filename_AISA, reproject=False )
    # reproject=False speeds up processing, otherwise each geometry would be individually reprojected prior to testing
    print("{:d} features (of total {:d}) found within raster {}".format( len(i_stand), len(geomlist_i), filename_AISA ) )

    # save to the big outlist all data which are inside the hyperspectral image
    for l,i in zip(outlist,outlist_i):
        l += [ i[ii] for ii in i_stand if not outlist_i[2][ii] in badstandlist ]

    # get the stand ids which are inside the AISA data
    #     this data has already been appended to outlist, 
    #     standids_i is for local use, looking up data in other tables
    standids_i = [ outlist_i[2][ii] for ii in i_stand if not outlist_i[2][ii] in badstandlist ]
    print("{:d} features left after excluding bad stands".format( len(standids_i) ) )

    # generate treestandid from standid. For the data modeled to beginning of 2018,  it should be 2000000000+standid
    tsids_i = [ i + 2000000000 for i in standids_i ]

    # Get all other required information and store it in outlist_extra
    # open the data table and keep it open for a while
    conn = sqlite3.connect( filename_mv )
    c = conn.cursor()
    
    listcounter = 0 # count how many elements are already in outlist_extra
    
    # -------- main species 
    # compute also and percentage of pine, spruce and broadleaf, these will be stored later
    #
    q=tools.hypdatatools_gdal.geopackage_getspecificvalues1(c,"treestandsummary","basalarea",tsids_i,"treestandid")
    # unpack data from list
    basalarea = [ i[0][0] if len(i)>0 else 0 for i in q ]
    q_species=tools.hypdatatools_gdal.geopackage_getspecificvalues1(c,"treestratum","treespecies",tsids_i,"treestandid")
    # q_species will be used later
    q_basal=tools.hypdatatools_gdal.geopackage_getspecificvalues1(c,"treestratum","basalarea",tsids_i,"treestandid")
    maintreespecies = []
    percentage_pine = []
    percentage_spruce = []
    percentage_broadleaf = []
    for areai,qi, q1i in zip( basalarea, q_basal, q_species ):
        if len(qi) > 0:
            basalareas = [ i[0] for i in qi ]
            if sum( basalareas ) > 0:
                species = [ i[0] for i in q1i ]
                uniquespecies = list( set(species) )
                basalarea_uniquespecies = [ sum( [ ba_k for ba_k,sp_k in zip(basalareas,species) if sp_k==sp_j ] ) for sp_j in uniquespecies ]
                sumbasalareas = sum(basalareas)
                # first, find main species. find all species present and find the basal area for each
                i_main = basalarea_uniquespecies.index( max(basalarea_uniquespecies) )
                basalarea_mainspecies = basalarea_uniquespecies[ i_main ] 
                maintreespecies.append( uniquespecies[ i_main ] )
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
                percentage_pine.append( 0 )
                percentage_spruce.append( 0 )
                percentage_broadleaf.append( 0 )
        else:
            maintreespecies.append( 0 )
            percentage_pine.append( 0 )
            percentage_spruce.append( 0 )
            percentage_broadleaf.append( 0 )
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( maintreespecies )
        outlist_extranames.append( "main_tree_species" )
    else:
        outlist_extra[listcounter] += maintreespecies
    listcounter += 1

    # -------- basalarea from treestandsummary [float] 
    q=tools.hypdatatools_gdal.geopackage_getspecificvalues1(c,"treestandsummary","basalarea",tsids_i,"treestandid")
    # unpack data from list
    basalarea = [ i[0][0] if len(i)>0 else 0 for i in q ]
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( basalarea )
        outlist_extranames.append( "basal_area" )
    else:
        outlist_extra[listcounter] += basalarea
    listcounter += 1
    
    # -------- dbh from treestandsummary [float]
    q=tools.hypdatatools_gdal.geopackage_getspecificvalues1(c,"treestandsummary","meandiameter",tsids_i,"treestandid")
    dbh = [ i[0][0] if len(i)>0 else 0 for i in q ]
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( dbh )
        outlist_extranames.append( "mean_dbh" )
    else:
        outlist_extra[listcounter] += dbh
    listcounter += 1 
    
    # -------- stem density from treestandsummary [integer]
    q=tools.hypdatatools_gdal.geopackage_getspecificvalues1(c,"treestandsummary","stemcount",tsids_i,"treestandid")
    stemcount = [ i[0][0] if len(i)>0 else 0 for i in q ]
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( stemcount )
        outlist_extranames.append( "stem_density" )
    else:
        outlist_extra[listcounter] += stemcount
    listcounter += 1

    # -------- mean height from treestandsummary [float] 
    q=tools.hypdatatools_gdal.geopackage_getspecificvalues1(c,"treestandsummary","meanheight",tsids_i,"treestandid")
    meanheight = [ i[0][0] if len(i)>0 else 0 for i in q ]
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( meanheight )
        outlist_extranames.append( "mean_height" )
    else:
        outlist_extra[listcounter] += meanheight
    listcounter += 1      

    # -------- percentage of pine, spruce and broadleaf [integer]
    #  these were calculated already earlier together with main tree species
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( percentage_pine)
        outlist_extranames.append( "percentage_of_pine" )
        outlist_extra.append( percentage_spruce )
        outlist_extranames.append( "percentage_of_spruce" )
        outlist_extra.append( percentage_broadleaf )
        outlist_extranames.append( "percentage_of_birch" )
    else:
        outlist_extra[listcounter+0] += percentage_pine
        outlist_extra[listcounter+1] += percentage_spruce
        outlist_extra[listcounter+2] += percentage_broadleaf
    listcounter += 3

    # -------- woody (branch+stem) biomass from treestandsummary [integer] 
    q1=tools.hypdatatools_gdal.geopackage_getspecificvalues1(c,"treestandsummary","stembiomass",tsids_i,"treestandid")
    q2=tools.hypdatatools_gdal.geopackage_getspecificvalues1(c,"treestandsummary","branchbiomass",tsids_i,"treestandid")
    stembiomass = [ i[0][0] if len(i)>0 else 0 for i in q1 ]
    branchbiomass = [ i[0][0] if len(i)>0 else 0 for i in q2 ]
    woodybiomass = [ round(sb+bb) for sb,bb in zip(stembiomass,branchbiomass) ]
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( woodybiomass )
        outlist_extranames.append( "woody_biomass" )
    else:
        outlist_extra[listcounter] += woodybiomass
    listcounter += 1

    # -------- LAI from treestratum [float] and effective LAI corrected for shootlevel clumping [float]
    # sum over all strata and use specific leaf weight for each species to get LAI
    q=tools.hypdatatools_gdal.geopackage_getspecificvalues1(c,"treestratum","leafbiomass",tsids_i,"treestandid")
    # assume leaf biomass is given as tons / ha = 1000 kg / 10,000 m2 = kg / (10 m2)
    #     = 100 g / m2
    LAI = []
    LAI_effective = []
    for spi,lbi in zip( q_species, q ):
        if len(spi) > 0:
            # replace all broadleaves (code 3 and above) with birch
            species = [ i[0] if i[0]<3 else 3 for i in spi ]
            lbiomass = [ i[0] for i in lbi ]
            if sum( lbiomass ) > 0:
                slw_i = [ slw[i] for i in species ] # specific leaf weight, g/m2
                STAR_i = [ STAR[i] for i in species ] 
                LAI_i = [ lb_j*100/slw_j for lb_j,slw_j in zip(lbiomass,slw_i) ]
                LAI_eff_i = [ 4*STAR[j]*LAI_j for j,LAI_j in zip(species,LAI_i) ]
                LAI.append( sum( LAI_i ) )
                LAI_effective.append( sum( LAI_eff_i ) )
            else:
                LAI.append( 0 )
                LAI_effective.append( 0 )
        else:
            LAI.append( 0 )
            LAI_effective.append( 0 )
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( LAI )
        outlist_extranames.append( "leaf_area_index" )
        outlist_extra.append( LAI_effective )
        outlist_extranames.append( "effective_leaf_area_index" )
    else:
        outlist_extra[listcounter] += LAI
        outlist_extra[listcounter+1] += LAI_effective
    listcounter += 2

    conn.close()

# =====================================================
# rasterize the data retrieved from FFC forestry data
# data in outlist and outlist_extra are processed once more, scaled if needed and stored as integer

# exclude first three colums (FID, geometries, standid) from saving as raster
#    add stand ID as the last layer
outfeatures = outlist[ 3: ] + outlist_extra + [outlist[2]]
outnames = fieldnames_in[ 1: ] + outlist_extranames + ['standID']

# YYY = XXX # stop before creating any files

# Discard border pixels: buffer the stands by -10 m
for i,j in enumerate( outlist[0] ):
    outlist[1][i] = outlist[1][i].Buffer(-10)

mvdata = tools.hypdatatools_img.create_raster_like( filename_AISA, filename_newraster , Nlayers=len(outfeatures), outtype=3, interleave='bip',
    force=True, description="Standlevel forest variable data from Metsakeskus geopackages" ) # outtypes 2=int, 3=long
mvdata_map = mvdata.open_memmap( writable=True )

# extra information is stored as json (but without outermost braces)
banddescriptions = {} # the big json dict

# create a memory shapefile with standid and fertility data
ii = 0 # Note to self: unclear why ii is necessary, why not use i_zip?
# loop over all data to be saved in the raster
for i_zip,(data,name) in enumerate(zip( outfeatures, outnames )):
    # convert data into a integer-encodable format        
    # create a memory shapefile with the field for rasterization
    descriptionlist = {} # description of the current band, to be used in banddescriptions
    valuelist = {} # list of possible values in a band, to be used in banddescriptions
    # do some necessary transformations for the data to store in integer format
    #  at the same time, create the extra metadata to store as json
    if name == "fertilityclass":
        name = "fertility_class" 
        data_converted = data
        descriptionlist['type'] = 'categorical'
        descriptionlist['values'] = { '0':'None', '1':'Herb-rich forest', '2':'Herb-rich heath forest',
            '3':'Mesic heath forest', '4':'Sub-xeric heath forest', '5':'Xeric heath forest',
            '6':'Barren heath forest'}
    elif name == "soiltype":
        print("Simplifying soil classification to 1:mineral/2:organic.")
        data_converted = [ 2 if (i>59 and i<70) else 1 for i in data ]
        name = "soil_class"
        descriptionlist['type'] = 'categorical'
        descriptionlist['values'] = {'0':'None', '1':'Mineral', '2':'Organic'}
    elif name == "main_tree_species":
        # change None to zero
        data_converted = [ i if i is not None else 0 for i in data  ]
        # merge "other broadleaves" (with values above 3) with birch (3)
        data_converted = [ i if i < 4 else 3 for i in data_converted  ]
        print(" converting other tree species to birch")
        # band name should end with "_class" for categorical variables
        name = "main_tree_species_class"
        descriptionlist['type'] = 'categorical'
        descriptionlist['values'] = {'0':'None', '1':'Scots pine', '2':'Norway spruce', '3':'birch and other broadleaves'}
    elif name == "basal_area":
        unitstring = 'm2/ha'
        storagefactor = 100 # factor for storage as int
        descriptionlist['type'] = 'continuous'
        descriptionlist['range'] = [0,None]
    elif name == "mean_height":
        storagefactor = 100 # factor for storage as int
        unitstring = 'm'
        descriptionlist['type'] = 'continuous'
        descriptionlist['range'] = [0,None]
    elif name == "mean_dbh":
        unitstring = 'cm'
        storagefactor = 100 # factor for storage as int
        descriptionlist['type'] = 'continuous'
        descriptionlist['range'] = [0,None]
    elif name == "leaf_area_index":
        unitstring = ''
        storagefactor = 100 # factor for storage as int
        descriptionlist['type'] = 'continuous'
        descriptionlist['range'] = [0,None]
    elif name == "effective_leaf_area_index":
        unitstring = ''
        storagefactor = 100 # factor for storage as int
        descriptionlist['type'] = 'continuous'
        descriptionlist['range'] = [0,None]
    elif name == "woody_biomass":
        unitstring = 't/ha'
        storagefactor = 1
        descriptionlist['type'] = 'continuous'
        descriptionlist['range'] = [0,None]        
    elif name == "stem_density":
        unitstring = '1/ha'
        storagefactor = 1 # factor for storage as int
        descriptionlist['type'] = 'continuous'
        descriptionlist['range'] = [0,None]
    elif name == "percentage_of_pine":
        unitstring = '%'
        storagefactor = 1 # factor for storage as int
        descriptionlist['type'] = 'continuous'
        descriptionlist['range'] = [0,100]
    elif name == "percentage_of_spruce":
        unitstring = '%'
        storagefactor = 1 # factor for storage as int
        descriptionlist['type'] = 'continuous'
        descriptionlist['range'] = [0,100]
    elif name == "percentage_of_birch":
        unitstring = '%'
        storagefactor = 1 # factor for storage as int
        descriptionlist['type'] = 'continuous'
        descriptionlist['range'] = [0,100]
    elif name == "standID":
        data_converted = data
        descriptionlist['type'] = 'categorical'
        descriptionlist['values'] = { '0':'N/A'}
        
    # complete data conversion and descriptionlist
    if descriptionlist['type'] == 'continuous':
        data_converted = [ int(i*storagefactor) for i in data ]
        if unitstring == '':
            unitstring_cat = ''
        else:
            unitstring_cat = "_[" + unitstring + "]"
        if storagefactor == 1:
            outnames[i_zip] = name+unitstring_cat
        else:
            outnames[i_zip] = name + "*" + str(storagefactor) + unitstring_cat
        descriptionlist['unit'] = [ 1/storagefactor , unitstring ]
    else:
        outnames[i_zip] = name

    descriptionlist['name'] = outnames[i_zip]
    banddescriptions[ descriptionlist['name'] ] = descriptionlist

    print("band {:d}: {} -- rasterizing... ".format( ii, name ), end ="" )
    memshp = tools.hypdatatools_gdal.vector_newfile( outlist[1], { name:data_converted } )
    # memraster = tools.hypdatatools_gdal.vector_rasterize_like( memshp, filename_AISA, shpfield=name, dtype=int )
    memraster = tools.hypdatatools_gdal.vector_rasterize_like( memshp, filename_AISA, shpfield=name, dtype="long" )
    # copy to envi file
    print(" saving... ", end="" )
    mvdata_map[:,:,ii] = memraster[:,:]
    print("done")
    ii += 1
    
# create buffers around hyperspectral image edges
# the current buffer size is 22 pixels, making it safe to use a 45x45 window in any area where forestry data exist
buffersize = 22
print("Creating mask for hyperspectral data")
hypdata = spectral.open_image( filename_AISA )
DIV = tools.hypdatatools_img.get_DIV(filename_AISA, hypdata=hypdata)
hypdata_map = hypdata.open_memmap()
# mask = np.all( hypdata_map==DIV, axis=2 )
# the above is very slow as it requires reading the whole file, use one band from the middle
mask = hypdata_map[:,:,100]==DIV 
# "no data" pixels are now masked out, for them mask==True
# expand the mask, so there would be no forestry data close to image edges
#   this way, when a window is used, the window would not include pixels with DIV (="no data")
mask_expanded = scipy.ndimage.maximum_filter( mask, buffersize, mode='constant', cval=True)
# the options mode='constant',cval=True make all areas outside the raster to be treated as DIV

# apply the mask
DIV_out = tools.hypdatatools_img.get_DIV( filename_newraster, hypdata=mvdata )
mvdata_map[ mask_expanded, : ] = DIV_out

mvdata.metadata['band names'] = outnames

# create banddescriptions and store to metadata
banddescriptions_sorted = {}
for i in outnames:
    # sort the dictionary for more readable output, assuming json string is in the same order as data are stored in the dict
    banddescriptions_sorted[i] = banddescriptions[i]
    # this will also create a nice error if not all data are described in banddescriptions

banddescriptions_str = json.dumps( banddescriptions_sorted )[1:-2] # exclude outermost braces
mvdata.metadata['band descriptions'] = banddescriptions_str

# close envi files, this will also save all changes to data (but apparently not metadata)
mvdata_map = None

# headers are apparently currently not updated by Spectral Python. Do it manually!
# tools.hypdatatools_img.envi_addheaderfield( filename_newraster, 'byte order', 0)
tools.hypdatatools_img.envi_addheaderfield( filename_newraster, 'band names', outnames )
tools.hypdatatools_img.envi_addheaderfield( filename_newraster, 'band descriptions', banddescriptions_str )

