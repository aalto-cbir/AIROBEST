"""
Copyright (C) 2017,2018  Matti Mõttus 
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Rasterizes Finnish Forestry Center standwise data

"""
import numpy as np
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import time
import os
import datetime

# load the hyperspectral functions -- not yet a package
#   add the folder with these functions to python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from tools.hypdatatools_img import *
from tools.hypdatatools_gdal import *
import tools.borealallometry

filenames_mv = [ "D:/mmattim/wrk/hyytiala-D/mv/MV_Juupajoki.gpkg", "D:/mmattim/wrk/hyytiala-D/mv/MV_Ruovesi.gpkg" ]

datafolder = 'D:/mmattim/wrk/hyytiala-D/AISA2017'
hyperspectral_filename = '20170615_reflectance_mosaic_128b.hdr' 

outfolder = 'D:/mmattim/wrk/hyytiala-D/temp'
filename_newraster = os.path.join( outfolder, "forestdata")

# Fill in species-specific STAR and slw
STAR = [None]*4
slw = [None]*4
for i in range( 1, 4 ):
    # STAR[0],slw[0] will remain None
    STAR[i] = tools.borealallometry.STAR( i )
    slw[i] = tools.borealallometry.slw( i )

# -----------------------------------------------------
# read metsäkeskus geopackages
# rasterize the selected data layers based on an already existing raster
#     existing raster is only used for defining the output geometry
# see code for details on which layers are rasterized
#   (the rasterized variables names area also printed out)

fieldnames_in = [ 'standid', 'fertilityclass', 'soiltype' , 'developmentclass' ] 
    # maintreespecies in treestandsummary is not trustworthy -- it is often empty


outlist = [ [], [] ] + [ [] for i in fieldnames_in ] # geometries and other data from the standid table
outlist_extra = [] # data combined from other data tables (list of lists)
outlist_extranames = [] # names of the variables in outlist_extra

for filename_mv in filenames_mv:
    outlist_i = ( vector_getfeatures( filename_mv, fieldnames_in ) )
    # outlist contains n+2 sublists: FID, geometries, standid, fertilityclass, etc
    geomlist_i = outlist_i[1]
    
    filename_AISA = os.path.join(datafolder,hyperspectral_filename)
    i_stand = geometries_subsetbyraster( geomlist_i, filename_AISA, reproject=False )
        # reproject=False speeds up processing, otherwise each geometry would be individually reprojected prior to testing
    print("{:d} features (of total {:d}) found within raster {}".format( len(i_stand), len(geomlist_i), filename_AISA ) )

    # save to the big outlist
    for l,i in zip(outlist,outlist_i):
        l += [ i[ii] for ii in i_stand ]

    # get the stand ids which are inside the AISA data
    #  this data has already been appended to outlist, standids_i is for local use, looking up datain other tables
    standids_i = [ outlist_i[2][i] for i in i_stand ]
    
    # open the data table and keep it open for a while
    conn = sqlite3.connect( filename_mv )
    c = conn.cursor()
    
    # -------  search for possible alterations in operations table
    # many of these are suggested operations with operationstate=0. (proposalyear, however, can be in the past) 
    # ignore these with operationstate=1
    q=geopackage_getspecificvalues1(c,"operation","completiondate",standids_i,"standid",additionalconstraint=" and operationstate = 1 ")
    # some plots have more than one operation. Get the latest
    #  XXX Can we be sure that the last is the latest? it may be best to convert text to date and check explicitly XXX 
    operationdate_latest = [ i[-1][0] if len(i)>0 else None for i in q ]
    
    listcounter = 0 # count how many elements are already in outlist_extra
    
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( operationdate_latest )
        outlist_extranames.append( "operationdate_latest" )
    else:
        outlist_extra[listcounter] += operationdate_latest
    listcounter += 1
        
    # -------- basalarea from treestandsummary [float] 
    # generate treestandid from standid. For the data modeled to beginning of 2018,  it should be 2000000000+standid
    tsids_i = [ i + 2000000000 for i in standids_i ]
    q=geopackage_getspecificvalues1(c,"treestandsummary","basalarea",tsids_i,"treestandid")
    #unpack data from list
    basalarea = [ i[0][0] if len(i)>0 else 0 for i in q ]
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( basalarea )
        outlist_extranames.append( "basalarea" )
    else:
        outlist_extra[listcounter] += basalarea
    listcounter += 1
    # -------- meanage from treestandsummary [integer]
    q=geopackage_getspecificvalues1(c,"treestandsummary","meanage",tsids_i,"treestandid")
    meanage = [ i[0][0] if len(i)>0 else 0 for i in q ]
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( meanage )
        outlist_extranames.append( "meanage" )
    else:
        outlist_extra[listcounter] += meanage
    listcounter += 1
    # -------- stemcount from treestandsummary [integer]
    q=geopackage_getspecificvalues1(c,"treestandsummary","stemcount",tsids_i,"treestandid")
    stemcount = [ i[0][0] if len(i)>0 else 0 for i in q ]
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( stemcount )
        outlist_extranames.append( "stemcount" )
    else:
        outlist_extra[listcounter] += stemcount
    listcounter += 1    
    # -------- meanheight from treestandsummary [float] 
    q=geopackage_getspecificvalues1(c,"treestandsummary","meanheight",tsids_i,"treestandid")
    meanheight = [ i[0][0] if len(i)>0 else 0 for i in q ]
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( meanheight )
        outlist_extranames.append( "meanheight" )
    else:
        outlist_extra[listcounter] += meanheight
    listcounter += 1    
    # -------- number of tree strata from treestratum [integer]
    q_basal=geopackage_getspecificvalues1(c,"treestratum","basalarea",tsids_i,"treestandid")
    # q_basal will also be used later
    N_strata = [ len(i) for i in q_basal ]
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( N_strata )
        outlist_extranames.append( "N_strata" )
    else:
        outlist_extra[listcounter] += N_strata
    listcounter += 1    
    # -------- main species, and percentage of mean species in basalarea from treestratum [integer]
    #                            percentages of pine, spruce and broadleaf [integer]
    #
    q_species=geopackage_getspecificvalues1(c,"treestratum","treespecies",tsids_i,"treestandid")
    # q_species will be used later
    percentage_mainspecies = []
    maintreespecies = []
    percentage_pine = []
    percentage_spruce = []
    percentage_broadleaf = []
    for areai,qi, q1i in zip( basalarea, q_basal, q_species ):
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
    # -------- woody (branch+stem) biomass from treestandsummary [integer] 
    q1=geopackage_getspecificvalues1(c,"treestandsummary","stembiomass",tsids_i,"treestandid")
    q2=geopackage_getspecificvalues1(c,"treestandsummary","branchbiomass",tsids_i,"treestandid")
    stembiomass = [ i[0][0] if len(i)>0 else 0 for i in q1 ]
    branchbiomass = [ i[0][0] if len(i)>0 else 0 for i in q2 ]
    woodybiomass = [ round(sb+bb) for sb,bb in zip(stembiomass,branchbiomass) ]
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( woodybiomass )
        outlist_extranames.append( "woodybiomass" )
    else:
        outlist_extra[listcounter] += woodybiomass
    listcounter += 1    
    # -------- LAI from treestratum [float] and effective LAI corrected for shootlevel clumping [float]
    # sum over all strata and use specific leaf weight for each species to get LAI
    q=geopackage_getspecificvalues1(c,"treestratum","leafbiomass",tsids_i,"treestandid")
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
        outlist_extranames.append( "LAI" )
        outlist_extra.append( LAI_effective )
        outlist_extranames.append( "LAI_effective" )
    else:
        outlist_extra[listcounter] += LAI
        outlist_extra[listcounter+1] += LAI_effective
    listcounter += 2
    
    # -------- dbh from treestandsummary [float]
    q=geopackage_getspecificvalues1(c,"treestandsummary","meandiameter",tsids_i,"treestandid")
    dbh = [ i[0][0] if len(i)>0 else 0 for i in q ]
    if len(outlist_extra) < listcounter + 1 :
        outlist_extra.append( dbh )
        outlist_extranames.append( "dbh" )
    else:
        outlist_extra[listcounter] += dbh
    listcounter += 1   

    conn.close()

# exclude first three colums (FID, geometries, standid) from saving as raster
outfeatures = outlist[ 3: ] + outlist_extra 
outnames = fieldnames_in[ 1: ] + outlist_extranames

# Discard border pixels: buffer the stands by -5 m
for i,j in enumerate( outlist[0] ):
    outlist[1][i] = outlist[1][i].Buffer(-5)

# rasterize the data retrieved from forestry data

mvdata = create_raster_like( filename_AISA, filename_newraster , Nlayers=len(outfeatures), outtype=3, interleave='bip',
    force=True, description="Standlevel forest variable data from Metsakeskus geopackages" ) # outtypes 2=int, 3=long
mvdata_map = mvdata.open_memmap( writable=True )

# create a memory shapefile with standid and fertility data
ii = 0
# loop over all data to be saved in the raster
for i_zip,(data,name) in enumerate(zip( outfeatures, outnames )):
    # convert data inoto a integer-encodable format
    
    # create a memory shapefile with the field for rasterization
    data_converted = data
    # do some naecessary transformations for the data to store in integer format
    if name == "developmentclass":
        # this includes strings
        unique_devclasses = [ i for i in set( data ) if i is not "" ] # exclude empty string
        unique_devclasses.insert( 0, "" ) #  insert empty string in the beginning, so it gets code 0
        codes_devclasses = [ i for i in range(len( unique_devclasses ) )  ]
        data_converted = [ codes_devclasses[ unique_devclasses.index( dd ) ] for dd in data ]
    elif name == "operationdate_latest":
        # save as integer from day 2000-1-1
        day0 = datetime.datetime(2000,1,1).toordinal()
        # nodatavalue = -32768
        nodatavalue = 0 # it might be confusing and if the completion was really on 1-1-2000 this creates some confusion. But 0 is the DIV of the file
        # replace 2000-01-01 2000-01-02 so not to have erroneous no data values
        if '2000-01-01' in data:
            data = [ '2000-01-02' if i=='2000-01-01' else i for i in data ]
        data_converted=[ datetime.datetime.strptime( i, '%Y-%m-%d' ).toordinal()-day0 if i is not None else nodatavalue for i in data ]
        outnames[i_zip] = name+"_[days_from_2000-01-01]"
    elif name == "maintreespecies":
        # change None to zero
        data_converted = [ i if i is not None else 0 for i in data  ]
        # merge "other broadleaves" (with values above 3) with birch (3)
        data_converted = [ i if i < 4 else 3 for i in data  ]
        print(" converting other tree species to birch")
    elif name == "basalarea":
        data_converted = [ int(i*100) for i in data ]
        outnames[i_zip] = name+"*100_[m2/ha]"
    elif name == "meanheight":
        data_converted = [ int(i*100) for i in data ]
        outnames[i_zip] = name+"_[cm]"        
    elif name == "dbh":
        data_converted = [ int(i*100) for i in data ]
        outnames[i_zip] = name+"*100_[cm]"
    elif name == "LAI":
        data_converted = [ int(i*100) for i in data ]
        outnames[i_zip] = name+"*100"
    elif name == "LAI_effective":
        data_converted = [ int(i*100) for i in data ]
        outnames[i_zip] = name+"*100" 
               
    print("band {:d}: {} -- rasterizing... ".format( ii, name ), end ="" )
    memshp = vector_newfile( outlist[1], { name:data_converted } )
    memraster = vector_rasterize_like( memshp, filename_AISA, shpfield=name, dtype=int )
    # copy to envi file
    print(" saving... ", end="" )
    mvdata_map[:,:,ii] = memraster[:,:]
    print("done")
    ii+=1

mvdata.metadata['band names'] = outnames

# close envi files, this will also save all changes to data (but apparently not metadata)
mvdata_map = None
# mvdata = None

# headers are apparently currently not updated by Spectral Python. Do it manually!
envi_addheaderfield( filename_newraster, 'band names', outnames )

