"""
Copyright (C) 2017,2018  Matti Mõttus 
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
"""
Some functions for working with hyperspectral data in ENVI file format
requires Spectral Python
various algorithms working on spectra (i.e., scientific stuff)
the functions here do not depend on GDAL.
"""
from tkinter import *
import numpy as np
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from scipy import stats
import os
import time
from scipy.optimize import curve_fit, least_squares

import tools.spectralinvariants

def p_processing( filename1, refspecno, wl_p, filename2, filename3, tkroot=None, file2_handle=None, file2_datahandle=None, progressvar=None ):
    """
    the actual function which does the processing
    inputs: 
      filename1 : reference data file name (incl full directory)
      refspecno : integer indicating which spectrum in filename1 to use
      wl_p : index of wavelengths used in computations 
      filename2 : the hyperspectraldata file which is used as input
      filename3 : the name of the data file to create
    optional inputs:
      tkroot : tkinter root window handle for signaling progress
      file2_handle=None : the spectral pyhton file handle if file is already open (for metadata)
      file2_datahandle=None : the spectral pyhton handle for hyperspectral data if file is already open
    filename2 is not reopened if the file handle exists (data handle is not checked)
      progressvar: NEEDS TO BE DoubleVar (of tkinter heritage) -- the variable used to mediate processing progress with a value  between 0 and 0.
        progressvar is also used to signal breaks by setting it to -1
    this function can be called separately from the commandline 
    """
    
    # read reference data and interpolate to hyperspectral bands 
    leafspectra = np.loadtxt(filename1)        
    # first column in leafspectra is wavelengths
    wl_spec = leafspectra[:,0]
    # which spectra should we use by default?
    refspec = leafspectra[ :, refspecno+1 ] # first column is wl, hence the "+1" 
        
    # get spectrum name from filename1
    # read first line assuming it contain tab-delimeted column headings
    with open(filename1) as f:
        refspectranames = f.readline().strip().split('\t')
        if refspectranames[0][0] != "#":
            # names not given on first line
            refspectranames = list(range(len(refspectranames)))
    refspecname = refspectranames[ refspecno+1 ] # first column is wl, hence the "+1"
    
    if file2_handle==None :
        # note:checking only the file handle. If file is opened, file2_datahandle is a matrix (and cannot be compared with None)
        # open hyperspectral data file -- reads only metadata
        hypdata = spectral.open_image(filename2)
        # hypdata.metadata is of type dict, use e.g. hypdata.metadata.keys()
        # e.g., print(hypdata.metadata.keys())
        # now get a handle for the data itself
        hypdata_map = hypdata.open_memmap()
    else:
        # the file is already open, use the provided handles
        hypdata = file2_handle
        hypdata_map = file2_datahandle
        print(filename2+" is already open, using the provided handles.")
    
    # wavelengths should be in metadata
    # if not, the program has no way to run, so let it crash
    wl_hyp = np.array(hypdata.metadata['wavelength'],dtype='float')
    if wl_hyp.max() < 100:
        # in microns, convert to nm
        wl_hyp *= 1000

    # interpolate refspec to hyperspectral bands
    # np.interp does not check that the x-coordinate sequence xp is increasing. If xp is not increasing, the results are nonsense. A simple check for increasing is:
    refspec_hyp = np.interp( wl_hyp, wl_spec, refspec )
    
    wl_p = np.array(wl_p) # tuples cannot be apparently used for indexing arrays
    # create a subset of reference data (interpolated to hyperspectral bands)
    refspec_hyp_subset = refspec_hyp[ wl_p ] # the reference spectrum subset to be used in calculations of p and DASF 

    # create metadata dictionary for p-file
    pfile_metadata = { 'bands':4, 'lines':hypdata.metadata['lines'], 'samples':hypdata.metadata['lines'],
        'data type':4, 'band names':['p','intercept','DASF','R']}
    # set data type to 4:float (32-bit); alternative would be 5:double (64-bit)
    # this needs to be synced with the envi.create_image command below
    pfile_metadata['description'] = "Created by python p-script from "+filename2
    pfile_metadata['description'] += "; reference spectrum " + refspecname
    pfile_metadata['description'] += "; WLs: " + ", ".join(['{:.2f}'.format(x) for x in wl_hyp[wl_p]])
    use_DIV = False # Use Data Ignore Value
    DIV = 0
    DIV_testpixel = wl_p[-1] # the band to check for data ignore value
    if 'data ignore value' in hypdata.metadata:
        pfile_metadata['data ignore value']=0 #use always zero for decimal compatibility
        use_DIV = True
        DIV = pfile_metadata['data ignore value']
    
    # check for optional keys to include
    for i_key in 'x start', 'y start', 'map info', 'coordinate system string':
        if i_key in hypdata.metadata:
            pfile_metadata[i_key] = hypdata.metadata[i_key]
    # create the data and open the file
    pfile = envi.create_image( filename3, metadata=pfile_metadata, dtype='float32', ext='', shape=hypdata_map.shape[0:2]+(4,) )
    pdata = pfile.open_memmap( writable=True )
    
    linecount = 1 # count lines for progress bar
    nlines = hypdata_map.shape[0]

    hypdata_factor = 1
    if int(hypdata.metadata['data type']) in (1,2,3,12,13):
        # these are integer data codes. assume it's reflectance*10,000
        hypdata_factor = 1.0 / 10000
    
    # set up timing
    t_0 = time.time()
    t_0p = time.process_time()

    break_signaled = False
    # iterate through hyperspectral data
    for hyp_line,p_line in zip(hypdata_map,pdata):
        if progressvar!=None:
            if progressvar.get() == -1: # check for user abort at each image line
                print("p_processing(): Break signaled")
                break_signaled = True
                break
            else:
                linecount += 1 # for updating the progress bar
                progressvar.set(linecount/nlines)
        else:
            print("#",end='')
        if not break_signaled:
            # for hyp_pixel,p_pixel in zip(hyp_line[0:50],p_line[0:50]):
            for hyp_pixel,p_pixel in zip(hyp_line,p_line):
                if hyp_pixel[ DIV_testpixel ] != DIV:
                    hyp_refl_subset = ( hyp_pixel[ wl_p ]*hypdata_factor )
                    # tools.spectralinvariants.p_forpixel_old( hyp_refl_subset, refspec_hyp_subset, p_pixel )
                    tools.spectralinvariants.p_forpixel( hyp_refl_subset, refspec_hyp_subset, p_pixel )
                else:
                    p_pixel.fill(0)

    # outer loop done: for hyp_line,p_line in zip(hypdata_map,pdata):
    # how long did it take?
    t_1 = time.time()
    t_1p = time.process_time()

    pdata.flush() # just in case, pdata will (likely?) be closed as function exits 
    if break_signaled:
        print("p_processing aborted at %4.2f%%." % (linecount/nlines*100) )
    else:
        print(" p_processing done")
    print( "time spent: " + str( round(t_1-t_0) ) + "s, process time: " + str( round(t_1p-t_0p)) + "s" )

def W_processing( filename1, filename2, filename3, DASFnumber=0, hypdata=None, hypdata_map=None, DASFdata=None, DASFdata_map=None, progressvar=None ):
    """
    the actual function which does the processing
    inputs: 
      filename1 : hyperspectral data file which is used as input (incl full directory)
      filename2 : the file with DASF data
      filename3 : the name of the data file to create
    optional inputs:
      DASFnumber = 0 : the band number in DASF file to use
      hypdata=None : the spectral pyhton file handle if file is already open (for metadata)
      hypdata_map=None : the spectral pyhton handle for hyperspectral data if file is already open
      DASFdata=None : the spectral pyhton file handle if DASF file is already open (for metadata)
      DASFdata_map=None : the spectral pyhton handle for DASF data if file is already open
      progressvar: NEEDS TO BE DoubleVar (of tkinter heritage) -- the variable used to mediate processing progress with a value  between 0 and 0.
        progressvar is also used to signal breaks by setting it to -1
      filename1 and filename2 are not reopened if the file handles exists (data handle is not checked)
    this function can be called separately from the commandline 
    """
    
    if hypdata is None:
        # open hyperspectral data file -- reads only metadata
        hypdata = spectral.open_image(filename1)
        # hypdata.metadata is of type dict, use e.g. hypdata.metadata.keys()
        # e.g., print(hypdata.metadata.keys())
        # now get a handle for the data itself
        hypdata_map = hypdata.open_memmap()
    else:
        # the file is already open, use the provided handles
        print(filename1 + " is already open, using the provided handles.")
    
    if DASFdata == None:
        DASFdata = spectral.open_image(filename2)
        DASFdata_map = DASFdata.open_memmap()
    else:
        print(filename2 + " is already open, using the provided handles.")
    
    # create metadata dictionary for W-file
    Wfile_metadata = { 'bands':hypdata_map.shape[2], 'lines':hypdata.metadata['lines'], 'samples':hypdata.metadata['lines'],
        'data type':12 }
    # set data type to 12: unsigned 16-bit int
    # this needs to be synced with the envi.create_image command below
    Wfile_metadata['description'] = "Spectral albedo W *10,000  created by python W-script from "+filename1+" and "+filename2
    
    use_DIV = False # Use Data Ignore Value
    DIV = 0
    # DIV_testpixel = 0 # the band used to test NO DATA
    DIV_testpixel = int(hypdata_map.shape[2]/2)  # the band used to test NO DATA: choose in the middle of the used wavelength range

    if 'data ignore value' in hypdata.metadata:
        Wfile_metadata['data ignore value']=0 # use always zero for decimal compatibility
        use_DIV = True
        DIV = float( Wfile_metadata['data ignore value'] )
    
    # check for optional keys to include    
    for i_key in 'map info', 'coordinate system string', 'wavelength':
        if i_key in hypdata.metadata:
            Wfile_metadata[i_key] = hypdata.metadata[i_key]
    
    # generate band names
    if 'wavelength' in hypdata.metadata:
        # W+wavelength
        bandnames = [ "W"+str(i) for i in hypdata.metadata['wavelength'] ]
    else:
        # consequtive numbers starting at 1
        bandnames = [ "W"+str(i) for i in range( 1,hypdata_map.shape[2]+1 ) ]

    # find the intersection of the to rasters (assumed to have the same resolution and to be in the same projection
    mapinfo_h = hypdata.metadata['map info']
    mapinfo_D = DASFdata.metadata['map info']
    # 0: projection name (UTM), 1: reference pixel x location (in file coordinates), 2: pixel y, 
    # 3: pixel easting, 4: pixel northing, 5: x pixel size, 6: y pixel size, 7: projection zone, North or South (UTM only)
    # In ENVI, pixel values always refer to the upper-left corner of the pixel
    dx_h = float( mapinfo_h[5] )
    dy_h = float( mapinfo_h[6] )
    ref_ix_h = float( mapinfo_h[1] ) # in image coordinates
    ref_iy_h = float( mapinfo_h[2] ) # in image coordinates
    ref_gx_h = float( mapinfo_h[3] ) # in geographic coordinates
    ref_gy_h = float( mapinfo_h[4] ) # in geographic coordinates
    lines_h = hypdata_map.shape[0]
    pixels_h = hypdata_map.shape[1]
    
    if 'x start' in hypdata.metadata.keys():
        xstart_h = int( hypdata.metadata['x start'] ) # the image coordinate for the upper-left hand pixel in the image
    else:
        xstart_h = 1
    if 'y start' in hypdata.metadata.keys():
        ystart_h = int( hypdata.metadata['y start'] )# the image coordinate for the upper-left hand pixel in the image
    else:
        ystart_h = 1

    # geographic coordinates of upper-left pixel corner of pixel (i,j) (i,j = 0,1,...N-1)
    #  gx = ref_gx + ( i + xstart - ref_ix )*dx
    #  gy = ref_gy - ( j + ystart - ref_iy )*dy (the image and geographic y-axes have opposite directions)
    #  i = ref_ix - xstart + ( gx - ref_gx )/dx 
    #  j = ref_iy - ystart - ( gy - ref_gy )/dy     
    
    # image extent in geographic coordinates
    xmin_h = ref_gx_h + ( xstart_h-ref_ix_h )*dx_h # i = 0
    xmax_h = ref_gx_h + ( pixels_h+xstart_h-ref_ix_h )*dx_h # lower-right corner
    ymax_h = ref_gy_h - ( ystart_h-ref_iy_h )*dy_h # envi image coordinates increase southwards (in negative y direction)
    ymin_h = ref_gy_h - ( lines_h+ystart_h-ref_iy_h )*dy_h # lower-right corner
    
    print( "Hyperspectral file: " + str(pixels_h) + "x" + str(lines_h) + " pixels" )

    # the same for DASF data
    dx_D = float( mapinfo_D[5] )
    dy_D = float( mapinfo_D[6] )
    ref_ix_D = float( mapinfo_D[1] ) # in image coordinates
    ref_iy_D = float( mapinfo_D[2] ) # in image coordinates
    ref_gx_D = float( mapinfo_D[3] ) # in geographic coordinates
    ref_gy_D = float( mapinfo_D[4] ) # in geographic coordinates
    lines_D = DASFdata_map.shape[0]
    pixels_D = DASFdata_map.shape[1]
    if 'x start' in DASFdata.metadata.keys():
        xstart_D = int( DASFdata.metadata['x start'] )
    else:
        xstart_D = 1
    if 'y start' in DASFdata.metadata.keys():
        ystart_D = int( DASFdata.metadata['y start'] )
    else:
        ystart_D = 1
    xmin_D = ref_gx_D + ( xstart_D-ref_ix_D )*dx_D # i = 0
    xmax_D = ref_gx_D + ( pixels_D+xstart_D-ref_ix_D )*dx_D # upper-left corner of lower-right pixel
    ymin_D = ref_gy_D - ( lines_D+ystart_D-ref_iy_D )*dy_D # upper-left corner of lower-right pixel
    ymax_D = ref_gy_D - ( ystart_D-ref_iy_D )*dy_D # envi image coordinates increase southwards (in negative y direction)
    
    print( "DASF file: " + str(pixels_D) + "x" + str(lines_D) + " pixels" )
        
    if dx_h != dx_D or dy_h !=dy_D:
        print("DASF and hyperspectral data have different resolutions. Failure is inevitable.")
    
    # for output file, in geographic coordinates
    xmin_o = max( [xmin_h,xmin_D] ) 
    xmax_o = min( [xmax_h,xmax_D] )
    ymin_o = max( [ymin_h,ymin_D] )
    ymax_o = min( [ymax_h,ymax_D] )
    # in image coordinates. Use hyperspectral file parameters
    imin_h = round( ref_ix_h - xstart_h + ( xmin_o - ref_gx_h )/dx_h ) 
    jmin_h = round( ref_iy_h - ystart_h - ( ymax_o - ref_gy_h )/dy_h ) 
    imax_h = round( ref_ix_h - xstart_h + ( xmax_o - ref_gx_h )/dx_h ) 
    jmax_h = round( ref_iy_h - ystart_h - ( ymin_o - ref_gy_h )/dy_h ) 
    pixels_o = imax_h - imin_h
    lines_o = jmax_h - jmin_h
    # calculate also the starting indices for DASF file. 
    imin_D = round( ref_ix_D - xstart_D + ( xmin_o - ref_gx_D )/dx_D )
    jmin_D = round( ref_iy_D - ystart_D - ( ymax_o - ref_gy_D )/dy_D )
    # DASF ending indices assuming equal pixel size:
    imax_D = imin_D + pixels_o
    jmax_D = jmin_D + lines_o
    
    print("Output file: " + str(pixels_o) + "x" + str(lines_o) + " pixels" )
    Wfile_metadata['x start'] = xstart_h+imin_h
    Wfile_metadata['y start'] = ystart_h+jmin_h
        
    # create the data and open the file
    bands = hypdata_map.shape[2]
    Wdata = envi.create_image( filename3, dtype='uint16', metadata=Wfile_metadata, ext='', shape=[ lines_o, pixels_o, bands ] )
    Wdata_map = Wdata.open_memmap( writable=True )
    
    linecount = 1 # count lines for progress bar

    # iterate through hyperspectral data
    hypdata_map_subset = hypdata_map[jmin_h:jmax_h,imin_h:imax_h,:]
    DASFdata_map_subset = DASFdata_map[jmin_D:jmax_D,imin_D:imax_D,DASFnumber]

    brange = range(bands)
    
    # set up timing
    t_0 = time.time()
    t_0p = time.process_time()

    break_signaled = False
    # for hyp_line,DASF_line,W_line in zip( hypdata_map[jmin_h:jmax_h,imin_h:imax_h,:], DASFdata_map[jmin_D:jmax_D,imin_D:imax_D,DASFnumber], Wdata_map):
    for hyp_line,DASF_line,W_line in zip( hypdata_map_subset, DASFdata_map_subset, Wdata_map):
        if progressvar is not None:
            if progressvar.get() == -1:
                break_signaled = True
                print("W_processing(): Break signaled.")
                break
            else:
                linecount += 1 # for updating the progress bar
                progressvar.set(linecount/lines_o)
        else:
            print("#",end='')

        if not break_signaled: # check for user abort at each image line
            for hyp_pixel,DASF_pixel,W_pixel in zip(hyp_line,DASF_line,W_line):
                if use_DIV:
                    if hyp_pixel[ DIV_testpixel ] != DIV:
                        # NOTE: test also the speed of zip and np.vectorize
                        # for ii in brange:
                        #    W_pixel[ii]  = round(hyp_pixel[ii]/DASF_pixel)
                        W_pixel[:] = hyp_pixel[:]/DASF_pixel
                        # NOTE: this algorithm is fast, but may induce rounding errors (when converting float->int)
                    else:
                        W_pixel.fill(0)
                else:
                    # for ii in brange:
                    #    W_pixel[ii]  = round( hyp_pixel[ii]/DASF_pixel )
                    W_pixel[:] = hyp_pixel[:]/DASF_pixel

    # outer loop done: for hyp_line,p_line in zip(hypdata_map,pdata):
    # how long did it take?
    t_1 = time.time()
    t_1p = time.process_time()
    
    Wdata_map.flush() # just in case, pdata will (likely?) be closed as function exits 
    if break_signaled:
        print("W_processing aborted at %5.2f%%." % (linecount/lines_o*100) )
    else:
        print("W_processing done")
    print( "time spent: " + str( round(t_1-t_0) ) + "s, process time: " + str( round(t_1p-t_0p)) + "s" )

def sbs_processing( hypfilename, hypdata=None, hypdata_map=None, paramvector=None, DIV=None, localprintcommand=None, progressvar=None ):
    """
    Compute the channel-specific Exemplar Score (ES) values for spectral band subsetting.
        Kang Sun, Xiurui Geng, and Luyan Ji
        Exemplar Component Analysis: A Fast Band Selection Method for Hyperspectral Imagery
        IEEE GEOSCIENCE AND REMOTE SENSING LETTERS, VOL. 12, NO. 5, MAY 2015
    Calls two subfunctions:
    1) sbs_distancematrix() which produces the matrix of between-band distances. This is the computer-intensive
        part of processing and utilizes input data. sbs_distancematrix() takes the same input parameters as 
        sbs_processing().
    2) sbs_ES() which computes ES values from distancematrix. This is fast and easy and requires no 
        hyperspectral data processing.
    
    input
        hypfilename: input filename. Not reopened, if hypdata is not None
        hypdata: the spectral python (SPy) file handle. If set, hypfilename is not reopened.
        hypata_map: the data raster which is extracted. If hypdata is set, hypfilename is not reopened.
        DIV: Data Ignore Value. If None, header is checked for its existence.
        paramvector: list of algorithm parameters [ sigmaparam, samplestep ]
            sigmaparam (sigma in Sun et al. 2015) is a adjusting parameter, controlling the weight 
                degradation rate as the function of distances.
            samplestep: integer determining how many pixels and lines to skip. If samplestep=10,
                every 10th pixel in every 10th row is used in computations. Default 1.
        localprintcommand: function to use for output. If None, print() will be used
        progressvar: NEEDS TO BE DoubleVar (of tkinter heritage) -- the variable used to mediate processing progress with a value between 0 and 0.
            progressvar is also used to signal breaks by setting it to -1
        
    returns: a numpy.ndarray with the length of the number of bands (ES values)
            or of length zero (in case of interruption)
    """
    
    ES = np.zeros(0) # output
    functionname = "sbs_processing() " # used in output messages
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    
    # call the first step
    # this is where hyperspectral data processing is done and can take a lot of time
    d_matrix = sbs_distancematrix( hypfilename, hypdata, hypdata_map, paramvector, DIV, localprintcommand, progressvar )

    if d_matrix.size > 0:
        # Proceed to the second step. This should be quick
        localprintcommand(functionname+" First step completed. Proceeding to computing ES.\n")
        ES = sbs_ES( d_matrix, paramvector=paramvector, localprintcommand=localprintcommand )
    else:
        # Processing was interrupted. Return empty matrix
        ES = np.array([])
    return ES
    
def sbs_distancematrix( hypfilename, hypdata=None, hypdata_map=None, paramvector=None, DIV=None, localprintcommand=None, progressvar=None ):
    """
    calculate the distance matrix used for Exempar Score (ES) values
        used by sbs_processing() and can be run manually.
        This is the time-consuming part of computations.
    input parameters the same as for sbs_ES()
        hypfilename: input filename. Not reopened, if hypdata is not None
        hypdata: the spectral python (SPy) file handle. If set, hypfilename is not reopened.
        hypata_map: the data raster which is extracted. If hypdata is set, hypfilename is not reopened.
        DIV: Data Ignore Value. If None, header is checked for its existence.
        paramvector: list of algorithm parameters [ sigmaparam, samplestep ]
            sigmaparam (sigma in Sun et al. 2015) is a adjusting parameter, controlling the 
                weight degradation rate as the function of distances -- NOT USED HERE
            samplestep: integer determining how many pixels and lines to skip. If samplestep=10,
                every 10th pixel in every 10th row is used in computations. Default 1.
        localprintcommand: function to use for output. If None, print() will be used
        progressvar: NEEDS TO BE DoubleVar (of tkinter heritage) -- the variable used to mediate processing progress with a value between 0 and 0.
            progressvar is also used to signal breaks by setting it to -1
        
    returns d: n_bands*n_bands numpy.ndarray with between-band distance metric (2-norm). d is symmetric with zeros on diagonal.
            in case of interruption, empty ndarray is returned
    """
    
    functionname = "sbs_distancematrix() " # used in output messages

    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    
    if hypdata is None:
        # open hyperspectral data file -- reads only metadata
        hypdata = spectral.open_image(hypfilename)
        # hypdata.metadata is of type dict, use e.g. hypdata.metadata.keys()
        # e.g., print(hypdata.metadata.keys())
        # now get a handle for the data itself
        hypdata_map = hypdata.open_memmap()
    else:
        # the file is already open, use the provided handles
        localprintcommand(functionname + hypfilename + " is already open, using the provided handles.\n")
    
    if DIV is None:
        use_DIV = False # Do not use Data Ignore Value
        DIV = 0    
        if 'data ignore value' in hypdata.metadata:
            use_DIV = True
            DIV = float( hypdata.metadata['data ignore value'] )
            localprintcommand(functionname + hypfilename + " has Data Ignore Value set to %i, using it.\n" % DIV )
    else:
        use_DIV = True
    DIV_testband = int(hypdata_map.shape[2]/2)  # the band used to test NO DATA: choose in the middle of the used wavelength range
    
    if paramvector is None:
        samplestep = 1
    else:
        samplestep = int(paramvector[1])

    if samplestep > 1:
        localprintcommand(functionname+" samplestep set, using every "+str(samplestep)+"th line and pixel.\n")

    lines_hyp = hypdata_map.shape[0]
    bands_hyp = hypdata_map.shape[2]
    
    d = np.zeros( (bands_hyp,bands_hyp) ) # OUTPUT, the distance matrix. The most time-consuming thing to compute

    elementcount = 1 # count elements in d for progress bar
    totalelements = bands_hyp*(bands_hyp-1)/2 # total independent elements in d
    progressincrement = 0.01 # the progress interval used in printing hash marks 
    nextreport = 0 # the next progress value at which to print a hash mark 
    
    # set up timing
    t_0 = time.time()
    t_0p = time.process_time()

    break_signaled_sbs = False
    # iterate through hyperspectral data   
    for bi in range(bands_hyp):
        if break_signaled_sbs:
            break
        # the distance rho is symmetric. Only compute the values for which j>i
        for bjj in range(0,bi):
            # the values for which j<i have already been computed. Fill them in.
            d[bi,bjj] = d[bjj,bi]
        for bj in range(bi+1,bands_hyp):
            # calculate d according to Eq. 1 in Sun et al. 2015
            d[bi,bj] = np.sqrt( np.sum( np.float64(hypdata_map[::samplestep,::samplestep,bi] - hypdata_map[::samplestep,::samplestep,bj])**2 ) )
            # check for breaks and calculate progress
            if progressvar is not None:
                if progressvar.get() == -1:
                    break_signaled_sbs = True
                else:
                    elementcount += 1 # for updating the progress bar
                    progressvar.set(elementcount/totalelements)
            else:
                progresstatus = elementcount/totalelements
                if progresstatus > nextreport:
                    localprintcommand("#")
                    nextreport += progressincrement

            if break_signaled_sbs: # check for user abort at each image line
                localprintcommand(functionname+"Break signaled at ")
                localprintcommand(" %i per cent complete.\n" % round(elementcount/totalelements*100) )
                break    
    
    # outer loop done: 
    # how long did it take?
    t_1 = time.time()
    t_1p = time.process_time()

    if break_signaled_sbs:
        localprintcommand( functionname + "aborted. time spent: " + str( round(t_1-t_0) ) + "s, process time: " + str( round(t_1p-t_0p)) + "s.\n" )
        d = np.array([])
    else:
        # normalize by the number of pixels
        # this is not required, but helps to compare different datasets
        # calculate the number of non-DIV cells (DIV cells contribute nothing to distance)
        if use_DIV:
            N = len( np.where( hypdata_map[::samplestep,::samplestep,DIV_testband] != DIV )[0] )
        else:
            hypdata_map_subset = hypdata_map[::samplestep,::samplestep,:]
            N = hypdata_map_subset.shape[0]*hypdata_map_subset.shape[1]
        d /= np.sqrt(N)
        localprintcommand( functionname + "finished (pixel skip " +str(samplestep) + "), time spent: " + str( round(t_1-t_0) ) + "s, process time: " + str( round(t_1p-t_0p)) + "s.\n" )
    return d

def sbs_ES( d_matrix, paramvector=None, localprintcommand=None ):
    """
    Second and final step in computing the Exemplar Score values for spectral band selection.
    This is the fast part of computations.
    input
        paramvector: list of algorithm parameters [ sigmaparam, samplestep ]
            sigmaparam is a adjusting parameter, controlling the weight degradation rate as the function of distances.
                
            samplestep: NOT USED HERE. integer determining how many pixels and lines to skip. 
        localprintcommand: function to use for output. If None, print() will be used
        
    returns: a numpy.ndarray with the length of the number of bands (ES values)

    """
    functionname = "sbs_ES() " # used in output messages

    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
        
    if paramvector is not None:
        sigmaparam = paramvector[0] # sigma in Sun et al. 2015
        if sigmaparam < 1:
            # not a useable value. the caller must hint that we should use the default.
            sigmaparam = None
    else:
        sigmaparam = None # set to none to signal that default is needed
        
    if sigmaparam is None:
        sigmaparam =sbs_defaultsigma( d_matrix )
        localprintcommand( functionname+"setting sigma to default value, "+str(sigmaparam)+".\n")

    bands_hyp = d_matrix.shape[0]

    rho = np.zeros( bands_hyp ) # rho as defined by Sun et al. 2015
    multfac = 1 / (2*sigmaparam**2) # the constant in the exponential in Eq. 2 in Sun et al. 2015
    for i in range(bands_hyp):
        rho[i] = sum( np.exp( -d_matrix[i,:]*multfac ) ) # Eq. 2 in Sun et al. 2015
    delta = np.zeros( bands_hyp ) # delta as defined by Sun et al. 2015
    for i in range(bands_hyp):
        jj = np.where( rho > rho[i] )[0]
        if len( jj ) == 0:
            delta[i] = np.max( d_matrix[i,:] ) # Eq. below #3, 2nd line in paragraph text, in Sun et al. 2015
        else:
            delta[i] = np.min( d_matrix[i,jj] ) # Eq. 3 in Sun et al. 2015
    ES = delta*rho # Eq. 4 in Sun et al. 2015

    localprintcommand( functionname + "finished.\n" )
    return ES

def sbs_defaultsigma( d_matrix ):
    # Default sigma = 1/30*mean(d) (Eq. 5 in Sun et al. 2015)
    sigmaparam = 1/30*np.mean(d_matrix)
    return sigmaparam

def PRI_processing( filename1, filename2, filename3, rhonumber=0, N=11, tkroot=None, hypdata=None, hypdata_map=None, rhodata=None, rhodata_map=None, progressvar=None ):
    """
    calculates the dependence of PRI on shadow fraction for an image
    according to the spectral invariant theory, sunlit fraction ~ intercept (denoted as rho)
    inputs: 
      filename1 : hyperspectral data file which is used as input (incl full directory)
      filename2 : the name with the intercept (='rho') data (for calculating sunlit fraction)
      filename3 : the name of the data file to create
    optional inputs:
      rhonumber =0 : the band number in intercept file to use
      N x N pixels are used for creating the relationship, N should be an odd number
      tkroot = None : tkinter root window handle for signaling progress
      hypdata=None : the spectral python file handle if file is already open (for metadata)
      hypdata_map=None : the spectral pyhton handle for hyperspectral data if file is already open
      rhodata=None : the spectral python file handle if intercept file is already open (for metadata)
      rhoFdata_map=None : the spectral python handle for intercept data if file is already open
      progressvar: NEEDS TO BE DoubleVar (of tkinter heritage) -- the variable used to mediate processing progress with a value between 0 and 0.
    filename1 and filename2 are not reopened if the file handles exists (data handle is not checked)
    this function can be called separately from the commandline 
    """
    
    minpixels = 9 # the minimum threshold for fitting data
    NDVIthreshold = 0.7 # the NDVI threshold for vegetated pixels
    
    if hypdata == None:
        # note:checking only the file handle. If file is opened, hypdata_map is a matrix (and cannot be compared with None)
        # open hyperspectral data file -- reads only metadata
        hypdata = spectral.open_image(filename1)
        # hypdata.metadata is of type dict, use e.g. hypdata.metadata.keys()
        # e.g., print(hypdata.metadata.keys())
        # now get a handle for the data itself
        hypdata_map = hypdata.open_memmap()
    else:
        # the file is already open, use the provided handles
        print(filename1 + " is already open, using the provided handles.")
    
    if rhodata == None:
        rhodata = spectral.open_image(filename2)
        rhodata_map = rhodata.open_memmap()
    else:
        print(filename2 + " is already open, using the provided handles.")
    
    # create metadata dictionary for output PRI-file
    PRIfile_metadata = { 'bands':4, 'lines':hypdata.metadata['lines'], 'samples':hypdata.metadata['lines'],
        'data type':4 }
    # set data type to 4:float (32-bit); alternative would be 5:double (64-bit)
    # this needs to be synced with the envi.create_image command below
    PRIfile_metadata['description'] = ( "PRI-rho regression created by python script PRI_sunlitfraction from " 
        + filename1+" and " + filename2 + ", window size " + str(N) + ", min(NDVI)=" + str( round(NDVIthreshold,3) ) )
    
    # use zero for data ignore value (DIV)
    DIV = 0
    PRIfile_metadata['data ignore value'] = DIV
    # existence of DIV is not checked in input files as zero pixels are masked out in NDVI mask
    # other DIV values are not used
    
    # check for optional keys to include    
    for i_key in 'map info', 'coordinate system string':
        if i_key in hypdata.metadata:
            PRIfile_metadata[i_key] = hypdata.metadata[i_key]
    
    # generate band names
    bandnames = [ 'PRI_0', 'k_570', 'k_531', 'cost' , 'N' ]
    PRIfile_metadata['band names'] = bandnames
    # in Markiet et al. 2017, the three parameters correspond to PRI_0, 0.2*Q_570, 0.2*Q_531
    # note the factor of 0.2, i.e. k = 0.2*Q. This factor comes from G and cos(t) and is included in model parameter here
    # cost: least_squares output, value of the cost function at the solution ( F(x) = 0.5 * sum(rho(f_i(x)**2), i = 0, ..., m - 1) )
    # N: the actual number of pixels used in this particular fit
    
    # find the intersection of the to rasters (assumed to have the same resolution and to be in the same projection
    mapinfo_h = hypdata.metadata['map info']
    mapinfo_r = rhodata.metadata['map info']
    # 0: projection name (UTM), 1: reference pixel x location (in file coordinates), 2: pixel y, 
    # 3: pixel easting, 4: pixel northing, 5: x pixel size, 6: y pixel size, 7: projection zone, North or South (UTM only)
    # In ENVI, pixel values always refer to the upper-left corner of the pixel
    dx_h = float( mapinfo_h[5] )
    dy_h = float( mapinfo_h[6] )
    ref_ix_h = float( mapinfo_h[1] ) # in image coordinates
    ref_iy_h = float( mapinfo_h[2] ) # in image coordinates
    ref_gx_h = float( mapinfo_h[3] ) # in geographic coordinates
    ref_gy_h = float( mapinfo_h[4] ) # in geographic coordinates
    lines_h = hypdata_map.shape[0]
    pixels_h = hypdata_map.shape[1]
    
    if 'x start' in hypdata.metadata.keys():
        xstart_h = int( hypdata.metadata['x start'] ) # the image coordinate for the upper-left hand pixel in the image
    else:
        xstart_h = 1
    if 'y start' in hypdata.metadata.keys():
        ystart_h = int( hypdata.metadata['y start'] )# the image coordinate for the upper-left hand pixel in the image
    else:
        ystart_h = 1

    # geographic coordinates of upper-left pixel corner of pixel (i,j) (i,j = 0,1,...N-1)
    #  gx = ref_gx + ( i + xstart - ref_ix )*dx
    #  gy = ref_gy - ( j + ystart - ref_iy )*dy (the image and geographic y-axes have opposite directions)
    #  i = ref_ix - xstart + ( gx - ref_gx )/dx 
    #  j = ref_iy - ystart - ( gy - ref_gy )/dy     
    
    # image extent in geographic coordinates
    xmin_h = ref_gx_h + ( xstart_h-ref_ix_h )*dx_h # i = 0
    xmax_h = ref_gx_h + ( pixels_h+xstart_h-ref_ix_h )*dx_h # lower-right corner
    ymax_h = ref_gy_h - ( ystart_h-ref_iy_h )*dy_h # envi image coordinates increase southwards (in negative y direction)
    ymin_h = ref_gy_h - ( lines_h+ystart_h-ref_iy_h )*dy_h # lower-right corner
    
    print( "Hyperspectral file: " + str(pixels_h) + "x" + str(lines_h) + " pixels" )

    # the same for intercept data
    dx_r = float( mapinfo_r[5] )
    dy_r = float( mapinfo_r[6] )
    ref_ix_r = float( mapinfo_r[1] ) # in image coordinates
    ref_iy_r = float( mapinfo_r[2] ) # in image coordinates
    ref_gx_r = float( mapinfo_r[3] ) # in geographic coordinates
    ref_gy_r = float( mapinfo_r[4] ) # in geographic coordinates
    lines_r = rhodata_map.shape[0]
    pixels_r = rhodata_map.shape[1]
    if 'x start' in rhodata.metadata.keys():
        xstart_r = int( rhodata.metadata['x start'] )
    else:
        xstart_r = 1
    if 'y start' in rhodata.metadata.keys():
        ystart_r = int( rhodata.metadata['y start'] )
    else:
        ystart_r = 1
    xmin_r = ref_gx_r + ( xstart_r-ref_ix_r )*dx_r # i = 0
    xmax_r = ref_gx_r + ( pixels_r+xstart_r-ref_ix_r )*dx_r # upper-left corner of lower-right pixel
    ymin_r = ref_gy_r - ( lines_r+ystart_r-ref_iy_r )*dy_r # upper-left corner of lower-right pixel
    ymax_r = ref_gy_r - ( ystart_r-ref_iy_r )*dy_r # envi image coordinates increase southwards (in negative y direction)
    
    print( "Intercept file: " + str(pixels_r) + "x" + str(lines_r) + " pixels" )

    if dx_h != dx_r or dy_h !=dy_r:
        print("Intercept and hyperspectral data have different resolutions. Failure is inevitable.")
    
    # image extent for output file
    # in geographic coordinates
    xmin_o = max( [xmin_h,xmin_r] ) 
    xmax_o = min( [xmax_h,xmax_r] )
    ymin_o = max( [ymin_h,ymin_r] )
    ymax_o = min( [ymax_h,ymax_r] )
    # in hyperspectral file coordinates
    imin_h = round( ref_ix_h - xstart_h + ( xmin_o - ref_gx_h )/dx_h ) # i stands for geographic x-coordinate, which the second index (index #1) in bsq files
    jmin_h = round( ref_iy_h - ystart_h - ( ymax_o - ref_gy_h )/dy_h ) # j stands for geographic y-coordinate, which the first index (index #0) in bsq files
    imax_h = round( ref_ix_h - xstart_h + ( xmax_o - ref_gx_h )/dx_h ) 
    jmax_h = round( ref_iy_h - ystart_h - ( ymin_o - ref_gy_h )/dy_h ) 
    pixels_o = imax_h - imin_h
    lines_o = jmax_h - jmin_h
    # calculate also the starting indices for intercept (rho) file. 
    imin_r = round( ref_ix_r - xstart_r + ( xmin_o - ref_gx_r )/dx_r ) 
    jmin_r = round( ref_iy_r - ystart_r - ( ymax_o - ref_gy_r )/dy_r )
    # Intercept ending indices assuming equal pixel size:
    imax_r = imin_r + pixels_o
    jmax_r = jmin_r + lines_o
    
    
    print("Output file: " + str(pixels_o) + "x" + str(lines_o) + " pixels" )
    PRIfile_metadata['x start'] = xstart_h+imin_h
    PRIfile_metadata['y start'] = ystart_h+jmin_h
        
    # create the data and open the file
    PRIdata = envi.create_image( filename3, dtype='float32', metadata=PRIfile_metadata, ext='', shape=[ lines_o, pixels_o, 5 ] )
    PRIdata_map = PRIdata.open_memmap( writable=True )
    
    linecount = 1 # count lines for progress bar

    # iterate through hyperspectral data
    hypdata_map_subset = hypdata_map[jmin_h:jmax_h,imin_h:imax_h,:]
    rhodata_map_subset = rhodata_map[jmin_r:jmax_r,imin_r:imax_r,rhonumber]

    dN = N//2 # max distance in pixels from area center
    
    # find the wavelengths closest to those in PRI and NDVI
    wl = np.array(hypdata.metadata['wavelength'],dtype='float')
    if wl.max() < 100:
        # in microns, convert to nm
        wl *= 1000
    i531 = ( (wl-531)**2 ).argmin()
    i570 = ( (wl-570)**2 ).argmin()
    i780 = ( (wl-780)**2 ).argmin()
    i680 = ( (wl-680)**2 ).argmin()
        
    # the model to fit (for scipy.optimize.curve_fit) 
    # def PRIfun(x, t):
    #   return t[0]- 0.5*np.log( (1+t[1]*x) / (1+t[2]*x ) )
    # Eq. 11 in Markiet et al. (2017)
    # parameters: 
    #  t[0]: PRI0
    #  t[1]: k*Q_570
    #  t[2]: k*Q_531
    #  x : sunlit fraction
    # Markiet et al. (2017) used k = 0.2 (a combination of G-function and cosine of solar angle).
    # the same model, for use with scipy.optimize.least_squares (a more flexible option)
    def PRIfun_lsq( t, x, y ):
        return t[0]- 0.5*np.log( (1+t[1]*x ) / (1+t[2]*x ) ) - y 
    #     # NOTE: scipy.optimize has also a leastsq() function??

    params_0 = [0.0,10.0,5.0] # initial parameter guesses (from Markiet et al., 2017 )

    print('Starting processing the whole image. This can be REALLY SLOW.')
    
    # set up time measurement
    t_0 = time.time()
    t_0p = time.process_time()

    break_signaled = False
    
    # find an area around the center pixel and fit the line to PRI-rho relationship
    for j_l in range(lines_o):
        j_l0 = max( 0, j_l-dN ) # the starting line of the area
        j_lend = min( lines_o, j_l+dN+1 ) # the last line of the area. +1 because of how python ranges are created
        if progressvar is not None:
            if progressvar.get()==-1: # check for user abort at each image line
                break_signaled = True
                print("Break signaled")
                break
            else:
                progressvar.set(j_l/lines_o)
        else:
            print('#',end='')

        for i_p in range(pixels_o):
            i_p0 = max( 0, i_p-dN ) # the starting pixel of the area
            i_pend = min( pixels_o, i_p+dN+1 ) # the last pixel of the area. +1 because of how python ranges are created

            R570 = hypdata_map_subset[ j_l0:j_lend,  i_p0:i_pend, i570 ]
            R531 = hypdata_map_subset[ j_l0:j_lend,  i_p0:i_pend, i531 ]
            R780 = hypdata_map_subset[ j_l0:j_lend,  i_p0:i_pend, i780 ]
            R680 = hypdata_map_subset[ j_l0:j_lend,  i_p0:i_pend, i680 ]          
            
            # exclude pixels where nan's may be produced (e.g., data ignore value, very dark pixels)
            den_PRI = (R531+R570).astype('float64')
            den_NDVI = (R780+R680).astype('float64')
            i1 = den_NDVI!=0
            den_NDVI[ i1 ] = ( R780[i1] - R680[i1] ) / den_NDVI[i1].astype('float') # den_NDVI contains zero if R760==-R780, NDVI for other pixels
        
            # three criteria fulfilled: denominators of PRI and NDVI not zero, and NDVI>0.8
            i_good = np.where( np.logical_and( den_NDVI > NDVIthreshold , den_PRI != 0 ) )
        
            if len(i_good[0]) > minpixels:
                PRI = ( R531[i_good] - R570[i_good] ) / den_PRI[i_good]
                rho = rhodata_map_subset[ j_l0:j_lend,  i_p0:i_pend ]
                rho = rho[ i_good ].astype('float64')
        
                # fit the curve
                res_lsq = least_squares( PRIfun_lsq, params_0, args=(rho, PRI), ftol=1e-12 )
                
                PRIdata_map[j_l,i_p,:] = [ res_lsq.x[0], res_lsq.x[1], res_lsq.x[2], res_lsq.cost, len(i_good[0]) ]

            else:
                PRIdata_map[j_l,i_p,:] = [ DIV, DIV, DIV, DIV, len(i_good[0]) ]
                
    # outer loop done: for j_l in range(lines_o):
    # how long did it take?
    t_1 = time.time()
    t_1p = time.process_time()
    print("")
    
    PRIdata_map.flush() # just in case, pdata will (likely?) be closed as function exits 
    if break_signaled:
        print("PRI_processing aborted at %4.1f%%" % (j_l/lines_o*100) )
    else:
        print(" done")
    print( "time spent: " + str( round(t_1-t_0) ) + "s, process time:" + str( round(t_1p-t_0p)) + "s" )

def PRI_singlepoint( filename1, filename2, x_h, y_h, N_area, rhonumber=0, hypdata=None, hypdata_map=None, rhodata=None, rhodata_map=None ):
    """
    the actual function which does the processing for one pixel only
    inputs: 
      filename1 : hyperspectral data file which is used as input (incl full directory)
      filename2 : the name with the intercept (='rho') data (for calculating sunlit fraction)
      x, y: image coordinates of the selected pixel in hypdata image
      # N_area x N_area pixels are used for creating the relationship, N_area should be an odd number
    optional inputs:    
      rhonumber =0 : the band number in intercept file to use
      hypdata=None : the spectral python file handle if file is already open (for metadata)
      hypdata_map=None : the spectral pyhton handle for hyperspectral data if file is already open
      rhodata=None : the spectral python file handle if intercept file is already open (for metadata)
      rhoFdata_map=None : the spectral python handle for intercept data if file is already open
    filename1 and filename2 are not reopened if the file handles exists (data handle is not checked)
    this function can be called separately from the commandline 
    """
    
    minpixels = 9 # the minimum threshold for fitting data
    NDVIthreshold = 0.7 # the NDVI threshold for vegetated pixels
    
    x_h = int( x_h )
    y_h = int( y_h )
    

    print('window '+str(N_area) )
    
    if hypdata == None:
        # note:checking only the file handle. If file is opened, hypdata_map is a matrix (and cannot be compared with None)
        # open hyperspectral data file -- reads only metadata
        hypdata = spectral.open_image(filename1)
        # hypdata.metadata is of type dict, use e.g. hypdata.metadata.keys()
        # e.g., print(hypdata.metadata.keys())
        # now get a handle for the data itself
        hypdata_map = hypdata.open_memmap()
    else:
        # the file is already open, use the provided handles
        print(filename1 + " is already open, using the provided handles.")
    
    if rhodata == None:
        rhodata = spectral.open_image(filename2)
        rhodata_map = rhodata.open_memmap()
    else:
        print(filename2 + " is already open, using the provided handles.")
    
    # find the geographic coordinates of the selected pixel
    mapinfo_h = hypdata.metadata['map info']
    # 0: projection name (UTM), 1: reference pixel x location (in file coordinates), 2: pixel y, 
    # 3: pixel easting, 4: pixel northing, 5: x pixel size, 6: y pixel size, 7: projection zone, North or South (UTM only)
    # In ENVI, pixel values always refer to the upper-left corner of the pixel
    dx_h = float( mapinfo_h[5] )
    dy_h = float( mapinfo_h[6] )
    ref_ix_h = float( mapinfo_h[1] ) # in image coordinates
    ref_iy_h = float( mapinfo_h[2] ) # in image coordinates
    ref_gx_h = float( mapinfo_h[3] ) # in geographic coordinates
    ref_gy_h = float( mapinfo_h[4] ) # in geographic coordinates
    lines_h = hypdata_map.shape[0]
    pixels_h = hypdata_map.shape[1]
    
    if 'x start' in hypdata.metadata.keys():
        xstart_h = int( hypdata.metadata['x start'] ) # the image coordinate for the upper-left hand pixel in the image
    else:
        xstart_h = 1
    if 'y start' in hypdata.metadata.keys():
        ystart_h = int( hypdata.metadata['y start'] )# the image coordinate for the upper-left hand pixel in the image
    else:
        ystart_h = 1

    # geographic coordinates of upper-left pixel corner of pixel (i,j) (i,j = 0,1,...N-1)
    #  gx = ref_gx + ( i + xstart - ref_ix )*dx
    #  gy = ref_gy - ( j + ystart - ref_iy )*dy (the image and geographic y-axes have opposite directions)
    #  i = ref_ix - xstart + ( gx - ref_gx )/dx 
    #  j = ref_iy - ystart - ( gy - ref_gy )/dy     
    
    # pixel in geographic coordinates
    x_g = ref_gx_h + ( x_h+xstart_h-ref_ix_h )*dx_h 
    y_g = ref_gy_h - ( y_h+ystart_h-ref_iy_h )*dy_h

    # find the coordinates of the selected pixel in intercept image
    mapinfo_r = rhodata.metadata['map info']
    dx_r = float( mapinfo_r[5] )
    dy_r = float( mapinfo_r[6] )
    ref_ix_r = float( mapinfo_r[1] ) # in image coordinates
    ref_iy_r = float( mapinfo_r[2] ) # in image coordinates
    ref_gx_r = float( mapinfo_r[3] ) # in geographic coordinates
    ref_gy_r = float( mapinfo_r[4] ) # in geographic coordinates
    lines_r = rhodata_map.shape[0]
    pixels_r = rhodata_map.shape[1]
    if 'x start' in rhodata.metadata.keys():
        xstart_r = int( rhodata.metadata['x start'] )
    else:
        xstart_r = 1
    if 'y start' in rhodata.metadata.keys():
        ystart_r = int( rhodata.metadata['y start'] )
    else:
        ystart_r = 1
        
    if dx_h != dx_r or dy_h !=dy_r:
        print("Intercept and hyperspectral data have different resolutions. Failure is inevitable.")
    
    # finally, x and y in rho image coordinates
    x_r = round( ref_ix_r - xstart_r + ( x_g - ref_gx_r )/dx_r )
    y_r = round( ref_iy_r - ystart_r - ( y_g - ref_gy_r )/dy_r )
    
    # assume identical resolution, hence constant shift between h and r images. Calculate the shift
    dx_hr = x_h - x_r
    dy_hr = y_h - y_r 
    # for any pixel, X_r = X_h - dx_hr; Y_r = Y_h - dy_hr
    
    # calculate image extents in hyperspectral data coordinates, so we know to select less pixels if we are close to edge
    #  start by finding extents in geographic coordinates
    # extent of hyperspectral image in geographic coordinates
    xmin_h = ref_gx_h + ( xstart_h-ref_ix_h )*dx_h # i = 0
    xmax_h = ref_gx_h + ( pixels_h+xstart_h-ref_ix_h )*dx_h # lower-right corner
    ymax_h = ref_gy_h - ( ystart_h-ref_iy_h )*dy_h # envi image coordinates increase southwards (in negative y direction)
    ymin_h = ref_gy_h - ( lines_h+ystart_h-ref_iy_h )*dy_h # lower-right corner
    # extent of intercept image in geographic coordinates
    xmin_r = ref_gx_r + ( xstart_r-ref_ix_r )*dx_r # i = 0
    xmax_r = ref_gx_r + ( pixels_r+xstart_r-ref_ix_r )*dx_r # upper-left corner of lower-right pixel
    ymin_r = ref_gy_r - ( lines_r+ystart_r-ref_iy_r )*dy_r # upper-left corner of lower-right pixel
    ymax_r = ref_gy_r - ( ystart_r-ref_iy_r )*dy_r # envi image coordinates increase southwards (in negative y direction)
    # overlapping area in geographic coordinates
    xmin_g = max( [xmin_h,xmin_r] ) 
    xmax_g = min( [xmax_h,xmax_r] )
    ymin_g = max( [ymin_h,ymin_r] )
    ymax_g = min( [ymax_h,ymax_r] )
    # overlapping area in hyperspectral image coordinates
    imin_h = round( ref_ix_h - xstart_h + ( xmin_g - ref_gx_h )/dx_h ) 
    jmin_h = round( ref_iy_h - ystart_h - ( ymax_g - ref_gy_h )/dy_h ) 
    imax_h = round( ref_ix_h - xstart_h + ( xmax_g - ref_gx_h )/dx_h ) 
    jmax_h = round( ref_iy_h - ystart_h - ( ymin_g - ref_gy_h )/dy_h ) 
    # overlapping area in intercept image coordinates
    imin_r = imin_h - dx_hr
    imax_r = imax_h - dx_hr
    jmin_r = jmin_h - dy_hr
    jmax_r = jmax_h - dy_hr

    dN = N_area//2 # max distance in pixels from area center
    
    # find the wavelengths closest to those in PRI and NDVI
    wl = np.array(hypdata.metadata['wavelength'],dtype='float')
    if wl.max() < 100:
        # in microns, convert to nm
        wl *= 1000
    i531 = ( (wl-531)**2 ).argmin()
    i570 = ( (wl-570)**2 ).argmin()
    i780 = ( (wl-780)**2 ).argmin()
    i680 = ( (wl-680)**2 ).argmin()
       
    # the model to fit (for scipy.optimize.curve_fit) 
    # def PRIfun(x, t):
    #   return t[0]- 0.5*np.log( (1+t[1]*x) / (1+t[2]*x ) )
    # Eq. 11 in Markiet et al. (2017)
    # parameters: 
    #  t[0]: PRI0
    #  t[1]: k*Q_570
    #  t[2]: k*Q_531
    #  x : sunlit fraction
    # Markiet et al. (2017) used k = 0.2 (a combination of G-function and cosine of solar angle).
    # the same model, for use with scipy.optimize.least_squares (a more flexible option)
    def PRIfun_lsq( t, x, y ):
        return t[0]- 0.5*np.log( (1+t[1]*x ) / (1+t[2]*x ) ) - y 
    #     # NOTE: scipy.optimize has also a leastsq() function??

    params_0 = [0.0,10.0,5.0] # initial parameter guesses (from Markiet et al., 2017 )

    # find an area around the center pixel and fit the line to PRI-rho relationship
    i0_h = max( imin_h, x_h-dN ) # the starting line of the area in hyperspectral data
    im_h = min( imax_h, x_h+dN+1 ) # the last line of the area. +1 because of how python ranges are created
    j0_h = max( jmin_h, y_h-dN ) # the starting pixel of the area
    jm_h = min( jmax_h, y_h+dN+1 ) # the last pixel of the area. +1 because of how python ranges are created    
    i0_r = i0_h - dx_hr
    im_r = im_h - dx_hr
    j0_r = j0_h - dy_hr
    jm_r = jm_h - dy_hr

    R570 = hypdata_map[ j0_h:jm_h, i0_h:im_h, i570 ]
    R531 = hypdata_map[ j0_h:jm_h, i0_h:im_h, i531 ]
    R780 = hypdata_map[ j0_h:jm_h, i0_h:im_h, i780 ]
    R680 = hypdata_map[ j0_h:jm_h, i0_h:im_h, i680 ]
        
    # exclude pixels where nan's may be produced (e.g., data ignore value, very dark pixels)
    den_PRI = (R531+R570).astype('float64')
    den_NDVI = (R780+R680).astype('float64')
    i1 = den_NDVI!=0
    den_NDVI[ i1 ] = ( R780[i1] - R680[i1] ) / den_NDVI[i1].astype('float') # den_NDVI contains zero if R760==-R780, NDVI for other pixels

    # three criteria fulfilled: denominators of PRI and NDVI not zero, and NDVI>0.8
    i_good = np.where( np.logical_and( den_NDVI > NDVIthreshold , den_PRI != 0 ) )

    if len(i_good[0]) > minpixels:
        PRI = ( R531[i_good] - R570[i_good] ) / den_PRI[i_good]
        rho = rhodata_map[ j0_r:jm_r, i0_r:im_r, rhonumber ]
        rho = rho[ i_good ].astype('float64')

        # fit the curve
        res_lsq = least_squares( PRIfun_lsq, params_0, args=(rho, PRI), ftol=1e-12 )
        
        # plot
        fig_PRIrho = plt.figure() # handle for spectrum figures
        fig_PRIrho.clf()
        
        ax_PRIrho = fig_PRIrho.add_subplot(1, 1, 1) # handle for the axes in refspec
        gx = np.linspace(0.0, max(rho) )
        gy = PRIfun_lsq( res_lsq.x, gx, np.zeros_like( gx ) )
        ax_PRIrho.plot( gx, gy, 'g-' )
        ax_PRIrho.plot( rho, PRI, 'rx' )
        ax_PRIrho.set_xlabel( 'rho' ) # XXX, the units should still be checked
        ax_PRIrho.set_ylabel( 'PRI' )
        ax_PRIrho.set_title( str(x_h) + ',' + str(y_h) + ' window ' + str(N_area) )

        fig_PRIrho.canvas.draw()
        fig_PRIrho.show()
        
        print('Fitting parameters for '+str(x_h) + ',' + str(y_h) + ' N=' + str(len(i_good[0])), end='') 
        print(', window size ' + str(N_area) + ', ' + str(round( min(den_NDVI[i_good]),3)) + '<NDVI<' + str(round(max(den_NDVI[i_good]),3))  )
        print( res_lsq.x )
    else:
        print( "Not sufficient green pixels after applying NDVI threshold of " + str(round(NDVIthreshold,3)) )
        print( "Pixels after thresholding = " + str(len(i_good[0])) + ", required " + str(minpixels) )

# 