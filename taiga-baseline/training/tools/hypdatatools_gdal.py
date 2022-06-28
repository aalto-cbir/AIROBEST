"""
The libraries for working with hyperspectral data in ENVI file format
requires Spectral Python
the functions here depend also on GDAL.
"""
import numpy as np
import spectral
import spectral.io.envi as envi
# from tkinter import filedialog
# from tkinter.scrolledtext import ScrolledText
from tkinter import *
import copy
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path 
import gdal
from osgeo import ogr,osr
from hypdatatools_img import *

# Mapping of Python types to OGR field types. Using other data types with this library will just fail
OGR_FIELD_TYPES_MAP = { int: ogr.OFTInteger, float: ogr.OFTReal, str: ogr.OFTString }
GDAL_FIELD_TYPES_MAP = { int: gdal.GDT_Int16, float: gdal.GDT_Float32, "long":gdal.GDT_Int32, "double":gdal.GDT_Float64  } # added string keys "long" and "double"


def get_rastergeometry( envihdrfilename, ignore_xystart=True, localprintcommand=None ):
    """
    Get the geometry (SpatialReference) of ENVI data file associated with envihdrfilename using GDAL
    outputs: SpatialReference, Geotransform
    envihdrfilename : the name with full path of ENVI .hdr file
    ignore_xystart: whether to ignore the 'x start' and 'y start' lines in the header file. GDAL ignores it,
        setting ignore_xystart=False would compensate for this ignorance. However, it seems that even files created by ENVI 
        have these set incorrectly (and 'x start' & 'y start' should be ignored)
    """
    
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
        
    functionname = "get_rastergeometry():" # for messaging
    
    filename1_data = envihdr2datafile( envihdrfilename, localprintcommand=localprintcommand )
    if filename1_data == '':
        localprintcommand( functionname + "get_rastergeometry(): Cannot find the data file corresponding to {}.\n".format(envihdrfilename) )
    f1_gdal = gdal.Open( filename1_data )
    f1_wkt = f1_gdal.GetProjection()
    f1_SpatialReference = osr.SpatialReference()
    f1_SpatialReference.ImportFromWkt(f1_wkt)
    f1_GeoTrans = f1_gdal.GetGeoTransform() # gives the same information as in "map info"
    # NOTE: it seems that GDAL ignores the "x start" and "y start" fields in ENVI header files
    # x start and y start define the image coordinates for the upper-left hand pixel in the
    # image. The default values are (1,1) so that the upper-left hand pixel has an image coordinate of (1,1).
    # use spectral to parse the header file
    hypdata = spectral.open_image( envihdrfilename )
    if ignore_xystart:
        xstart = 1
        ystart = 1
    else:
        if 'x start' in hypdata.metadata:
            xstart = int( hypdata.metadata['x start'] )
        else:
            xstart = 1 # default value
        if 'y start' in hypdata.metadata:
            ystart = int( hypdata.metadata['y start'] )
        else:
            ystart = 1 # default value
    startvalues = ( xstart, ystart )
    
    # (GT=GeoTrans): coefficients for transforming between pixel/line (P,L) raster space, and projection coordinates (Xp,Yp) space
    # Xp = GT[0] + P*GT[1] + L*GT[2];
    # Yp = GT[3] + P*GT[4] + L*GT[5];
    # The inverse in the general case is  
    # P = ( Xp*GT[5] - GT[0]*GT[5] + GT[2]*GT[3] - Yp*GT[2] ) / ( GT[1]*GT[5] - GT[2]*GT[4] )
    # L = ( Yp*GT[1] - GT[1]*GT[3] + GT[0]*GT[4] - Xp*GT[4] ) / ( GT[1]*GT[5] - GT[2]*GT[4] )
    # NOTE: ENVI files refer to pixels by their upper-left corner. It is more convenient to use pixel center coordinates
    #   if the coordinates p and l are given relative to pixel centers, we get
    # Xp = GT[0] + (p+0.5)*GT[1] + (l+0.5)*GT[2];
    # Yp = GT[3] + (p+0.5)*GT[4] + (l+0.5)*GT[5];
    # D = GT[1]*GT[5] - GT[2]*GT[4]
    # p = ( Xp*GT[5] - GT[0]*GT[5] + GT[2]*GT[3] - Yp*GT[2] ) / D - 0.5
    # l = ( Yp*GT[1] - GT[1]*GT[3] + GT[0]*GT[4] - Xp*GT[4] ) / D - 0.5
    #
    # Finally, x start and y start values need to be applied separately.
    #
    # print("Raster has geometry " + f1_SpatialReference.ExportToProj4() )
    return f1_SpatialReference, f1_GeoTrans, startvalues
    
def world2image( envihdrfilename, pointmatrix ):
    """
    convert the (usually projected) world coordiates in pointmatrix to the image 
    coordinates of envihdrfilename (relative to pixel center).
    pointmatrix: 2-column np.matrix [[x, y]]
        NOTE: np.ndarray will not allow consistent indexing
    """
    SR_r, GT, startvalues = get_rastergeometry( envihdrfilename )

    # transform to hyperspectral figure coordinates
    X = pointmatrix[:,0]
    Y = pointmatrix[:,1]
    D = GT[1]*GT[5] - GT[2]*GT[4]
    xy = np.column_stack( ( ( X*GT[5] - GT[0]*GT[5] + GT[2]*GT[3] - Y*GT[2] ) / D - 0.5 ,
         ( Y*GT[1] - GT[1]*GT[3] + GT[0]*GT[4] - X*GT[4] ) / D - 0.5 ) )
    xy[:,0] -= ( startvalues[0] - 1 )
    xy[:,1] -= ( startvalues[1] - 1 )
    return xy
    
def image2world ( envihdrfilename, pointmatrix_local ):
    """
    convert the image coordinates (relative to pixel center) in pointmatrix_local to the 
    (usually projected) world coordinates of envihdrfilename.
    pointmatrix_local: 2-column np.matrix [[x, y]]
    """
    SR_r, GT, startvalues = get_rastergeometry( envihdrfilename )

    # transform to hyperspectral figure coordinates
    P = pointmatrix_local[:,0] + (startvalues[0]-1) + 0.5 # relative to pixel corner
    L = pointmatrix_local[:,1] + (startvalues[1]-1) + 0.5
    xy = np.column_stack(  ( GT[0] + P*GT[1] + L*GT[2],
                    GT[3] + P*GT[4] + L*GT[5] ) )

    return xy 
    
def shape2imagecoords( geometry, hypfilename, localprintcommand=None ):
    """
    Convert a vector to the image coordinates of a raster
    input: 
        geometry: ogr.Geometry with points (e.g., a ring in a POLYGON)
        hypfilename: hyperspectral data file name
        localprintcommand: the local routine for message output. It is not used, only passed through
    output:
        xy: 2-column numpy matrix of points in hypfilename image geometry
    """

    # make a clone of the geometry so the original would not be transformed
    C = geometry.Clone()

    if C.GetGeometryName() == "POLYGON":
        # we need to get the outer ring which contains points.
        #   Note: this ring (also a ogr.Geometry) lacks spatial reference -- get this from the polygon
        #   Some other geometries may also be potentially useful, see ??8.2.8 (page 66) of
        #   http://portal.opengeospatial.org/files/?artifact_id=25355
        C = geometry.GetGeometryRef(0) # not to have side effects
        if C.GetSpatialReference() is None:
            C.AssignSpatialReference( geometry.GetSpatialReference() )

    SR_r, GT, startvalues = get_rastergeometry( hypfilename, localprintcommand=localprintcommand )
    # print( "raster: "+SR_r.ExportToProj4() ) # XXX 

    SR_v = C.GetSpatialReference()
    # print( "vector: "+SR_v.ExportToProj4() )
    # transform vector to raster coordinates
    vr_transform = osr.CoordinateTransformation( SR_v, SR_r )
    C.Transform( vr_transform )
    xy_w = np.array( C.GetPoints() ) # points in world coordinates
    # transform to hyperspectral image coordinates
    # D = GT[1]*GT[5] - GT[2]*GT[4]
    # xy = np.array([ (xy_w[:,0]*GT[5] - GT[0]*GT[5] + GT[2]*GT[3] - xy_w[:,1]*GT[2] ) / D - 0.5 ,
    #         ( xy_w[:,1]*GT[1] - GT[1]*GT[3] + GT[0]*GT[4] - xy_w[:,0]*GT[4] ) / D - 0.5] ).transpose()
    xy_i = world2image( hypfilename, xy_w )
    return xy_i
    
def loadpolygon( filename_shape , localprintcommand=None ):
    """
    Load the polygons from shapefile into a structure in memory
    returns a list of gdal geometries
    """
    
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
        
    functionname = "loadshp_fun():" # for messaging
    polygonlist = []
    
    if len( self.polygonlist ) > 0:
        localprintcommand( functionname + " Some polygons already exist. Deleting.\n")
        self.polygonlist = []
    sh_file = ogr.Open( filename_shape )
    
    # get some information and check for validity
    N_layers = sh_file.GetLayerCount() 
    localprintcommand( functionname + " Shapefile: " + str( N_layers ) + " layers")
    N_pts = 0 # the number of points in the final polygon. Used also to test if a polygon has been found
    # find the first layer with some features
    for il in range(N_layers):
        sh_layer = sh_file.GetLayerByIndex(il)
        # print( sh_layer.GetExtent() )
        sh_SpatialReference = sh_layer.GetSpatialRef()
        # sh_f = sh_layer.GetFeature(0) #  The returned feature should be free with OGR_F_Destroy(). -- not done in Cookbook?
        sh_f = sh_layer.GetNextFeature() 
        while sh_f != None:
            sh_g = sh_f.GetGeometryRef()
            geom_clone = sh_g.Clone() # store a clone so the file can be (hopefully) closed

            # try to make sure we have SpatialReference set. It seems to be lost sometimes
            if geom_clone.GetSpatialReference() is None and sh_SpatialReference!=None:
                # XXX is this really necessary? Can it happen that a feature would not have SpatialReference?
                geom_clone.AssignSpatialReference( sh_SpatialReference )                 
            polygonlist.append( geom_clone )
            sh_f = sh_layer.GetNextFeature() 
            
        localprintcommand(", layer " + str(il) + ", "+ sh_layer.GetName() + ", has " + str(sh_layer.GetFeatureCount()) + " feature(s).\n" )
        N_pts = geom_clone.GetPointCount()
        if geom_clone.GetGeometryName() == 'POLYGON':
            # NOTE: some other geometries may also be potentially useful, see ??8.2.8 (page 66) of
            #   http://portal.opengeospatial.org/files/?artifact_id=25355
            # Points of polygon cannot be accessed directly, we need to get to the geometry first
            geom_clone_ring = geom_clone.GetGeometryRef(0)
            N_pts = geom_clone_ring.GetPointCount()
        localprintcommand( functionname + " Last geometry is of type " + geom_clone.GetGeometryName()  + " with " + str(N_pts) + " points.\n")
        # localprintcommand( "geometry : "+ R.GetSpatialReference().ExportToProj4() + ".\n" )
                    
    if N_pts == 0:
        localprintcommand(functionname + " Could not load shapefile, no suitable features found.\n")
    return polygonlist
    
def plot_vector( figurehandle, figuredatafile, geometry, color='r' ):
    """
    Plot the vector points in the geometry in the matplotlib raster windows already containing file figuredatafile. Reprojects XY
    in:
        figurehandle: matplotlib figure handle, matplotlib.figure.Figure
        figuredatafile: file name of the raster plotted in the figure. Required to get figure coordinates.
        geometry is a polygon or the "ring" of a polygon, of type osgeo.ogr.Geometry
        color is a color in a format accepted by matplotlib
    """
    
    # if localprintcommand is None:
    #     # use a print command with no line feed in the end. The line feeds are given manually when needed.
    #     localprintcommand = lambda x: print(x,end='')

    if geometry.GetGeometryName() == "MULTIPOLYGON":
        # call the function recursively for each sub-POLYGON
        for i in range( geometry.GetGeometryCount() ):
            # make sure the spatial reference is passed on
            if  geometry.GetGeometryRef(i).GetSpatialReference() is None:
                subgeometry = geometry.GetGeometryRef(i).Clone()
                subgeometry.AssignSpatialReference( geometry.GetSpatialReference() )
            else:
                subgeometry = geometry.GetGeometryRef(i)
            plot_vector( figurehandle, figuredatafile, subgeometry, color=color )
    else:
        if geometry.GetGeometryName() == "POLYGON":
            # we need to get the outer ring which contains points.
            #   Note: this ring (also a ogr.Geometry) lacks spatial reference -- get this from the polygon
            geom_in = geometry
            geometry = geometry.GetGeometryRef(0)
            if geometry.GetSpatialReference() is None:
                geometry.AssignSpatialReference( geom_in.GetSpatialReference() )
        # we make the assumption now that if the geometry now refers to a proper outer ring of a POLYGON.
        # documentation: OpenGIS simple features polygons consist of one outer ring (linearring), and zero or more inner rings
        
        xy = shape2imagecoords( geometry, figuredatafile )    
        figurehandle.axes[0].plot( xy[:,0], xy[:,1], c=color )
        
    figurehandle.canvas.draw()

    
def plot_clearvectors( figurehandle ):
    """
    clear the figure with figure handle of vector drawings
    in:
        figurehandle: matplotlib figure handle, matplotlib.figure.Figure
    """

    while len( figurehandle.axes[0].get_lines() ) > 0:
        figurehandle.axes[0].get_lines()[0].remove()
    figurehandle.canvas.draw()

  
def pixel_coords( hypfilename, point, areasize, areaunit, areashape, hypdata=None ):
    """
    give the coordinates of pixels (in image coordinates) to be sampled around point
    hypfilename: the filename or spectral file  handle of hyperspectral data file
    point: (x,y) coordinates of the point in global coordinates
    areasize: size of the sampled area (circle diameter, square side)
    areaunit: unit in which areasize is given ("meter" or "pixel", first letter suffices)
    areashape: 'circle' or 'square' (first letter suffices)
    hypdata, spectral handle for file the file. If given, hypfilename will not be reopened
    output
    coordlist: list of two lists: ( (y) , (x) )
     NOTE! Envi BIL files have y (line) for first coordinate [0], x (pixel) for second [1]
    """
    if hypdata is None:
        # open the file if not open yet. This only gives access to metadata.                
        hypdata = spectral.open_image( hypfilename )
        # open the file as memmap to get the actual hyperspectral data
        print("opening file "+hypfilename)
        if hypdata.interleave == 1:
            print("Band interleaved (BIL)")
        else:
            print( hypfilename + " not BIL -- opening still as BIL -- will be slower" )
    
    lines = hypdata.nrows
    pixels = hypdata.ncols

    if areaunit == 'meters':
        # find out what one pixel means in meters in each image direction
        # assume GT[0]==GT[3]==0 so that origins match and move by one pixel in either direction
        SR_r, GT, startvalues = get_rastergeometry( hypfilename )
        di = np.sqrt( GT[1]**2 + GT[4]**2 )
        dj = np.sqrt( GT[2]**2 + GT[5]**2 )
        # calculate conversion factors: pixels = ptom * meters
        ptom_i = 1.0 / di
        ptom_j = 1.0 / dj
    else:
        ptom_i = 1 # integer one
        ptom_j = 1
    
    xy = world2image( hypfilename, np.matrix(point) ) # a row matrix
    
    if areaunit[0]=='p' and areasize//2 != areasize/2 and areashape[0]=='s' :
        # odd number of pixels, choose symmetrically around center
        imin = int( round( xy[0,0] - areasize/2 ) )
        jmin = int( round( xy[0,1] - areasize/2 ) )
    else:
        # even or non-integer number of pixels, the general case
        imin = int( round( xy[0,0] )  - (areasize*ptom_i)//2 )
        jmin = int( round( xy[0,1] )  - (areasize*ptom_i)//2 )
    imax = max( imin, imin + int( round( areasize*ptom_i ) ) - 1 )
    jmax = max( jmin, jmin + int( round( areasize*ptom_j ) ) - 1 )
    if imin < pixels and imax > -1 and jmin < lines and jmax > -1:
        imin = max( imin, 0 )
        jmin = max( jmin, 0 )
        imax = min( imax, pixels-1 ) # '-1' accounts for numbering starting at zero
        jmax = min( jmax, lines-1 )
        # print( imin,jmin,imax,jmax) # XXX
        # get the indices of points and store in an array
        coordlist_x = []
        coordlist_y = []
        if areashape[0] == 's':
            # square, do not test for distance, set it to infinity
            maxdist2 = float('inf')
        else:
            maxdist2 = (areasize/2)**2 # max squared distance from center pixel
        for i in range( imin, imax+1 ):
            dx2 = np.square( ( i - xy[0,0] )*ptom_i )
            for j in range( jmin, jmax+1 ):
                dy2 = np.square( ( j - xy[0,1] )*ptom_j )
                if dx2 + dy2 <= maxdist2:
                    coordlist_x.append( i )
                    coordlist_y.append( j )
        coordlist = [ coordlist_y , coordlist_x ]

    else:
        print("pixel_coords(): point out of image:", end='')
        print(point)
        coordlist = []

    return coordlist

def points_from_shape( rasterfile_in, geometry, rasterfile_hdr=None, localprintcommand=None ):
    """ 
    subset a raster file with a vector using matplotlib.path
    outputs the coordlist [ [x0], [x1] ] ] of the coordinates in points inside the raster file in global coordinates
        the coordlist can be directly used to subset a raster
    inputs
        rasterfile_in: name of hyperspectral data file, the header file, or GDAL raster file handle (gdal.Dataset)
        geometry: ogr.Geometry, currently works with MULTIPOLYGONs, POLYGONs or the LINEARRINGs inside them
            OpenGIS simple features polygons consist of one outer ring (linearring), and zero or more inner rings.
            NOTE: the linearrings may not have a SpatialRef attached.
            Some other geometries may also be potentially useful, see ??8.2.8 (page 66) of
            http://portal.opengeospatial.org/files/?artifact_id=25355
        rasterfile_hdr: the enf´vi header file name. If given, rasterfile_in is assumed to be GDAL raster file handle (gdal.Dataset) without any checks
        localprintcommand: the local routine for message output. It is not used, only passed through
    output
        coordlist [ [x0], [x1] ] ]: a list of two lists, each containing the coordinates of the points in image coordinates
    NOTE: the extent of the geometry is calculated assuming no rotation between global and image coordinate systems
    """
    
    if rasterfile_hdr is not None:
        # assume that rasterfile_envihdr is a string ontaining ENVI hdr file name and 
        #  rasterfile_in is GDAL raster file handle (gdal.Dataset)
        rasterfile = rasterfile_in
        
    elif type( rasterfile_in ) is gdal.Dataset:
        rasterfile = rasterfile_in
        rasterfile_hdr = datafile2envihdr( rasterfile.GetDescription(), localprintcommand=localprintcommand ) 
            # get the .hdr filename for envi; GetDescription() gives the name of the data file
            # we need both the GDAL and SpectralPython access to data, decoding the name each time is necessary evil
            #   it takes a lot of time, but I see no way around it (passing two arguments is an option now, though)
    else:
        # rasterfile_in is either envi hdr or envi datafile name
        rasterfile, rasterfile_hdr = envifilecomponents( rasterfile_in, localprintcommand=localprintcommand )
        rasterfile = gdal.Open( rasterfile )
    
    if geometry.GetGeometryName() == "MULTIPOLYGON":
        # MULTIPOLYGONS consist of multiple subpolygons
        # use recursion to retrieve the points from the subpolygons
        coordlist = [[],[]] # output list where all subplygons are merged
        for i in range( geometry.GetGeometryCount() ):
            # the subpolygons don't (usually?) have SpatialReference set, hence we need 
            #  to set it, and modify our input argument (which is not desired here). Hence, clone        
            if  geometry.GetGeometryRef(i).GetSpatialReference() is None:
                subgeometry = geometry.GetGeometryRef(i).Clone()
                subgeometry.AssignSpatialReference( geometry.GetSpatialReference() )
            else:
                subgeometry = geometry.GetGeometryRef(i)
            subcoordlist = points_from_shape( rasterfile, subgeometry, rasterfile_hdr=rasterfile_hdr, localprintcommand=localprintcommand )
            coordlist = [ coordlist[0]+subcoordlist[0] , coordlist[1]+subcoordlist[1] ]
    else:
        # the actual work, substract points
        # convert the outer ring of the geometry from a GDAL polygon to a matplotlib path
        if geometry.GetGeometryName() == "POLYGON":
            geom_in = geometry
            geometry = geometry.GetGeometryRef(0)
            if geometry.GetSpatialReference() is None:
                geometry.AssignSpatialReference( geom_in.GetSpatialReference() )
        # we make the assumption now that if the geometry now refers to a proper outer ring of a POLYGON.
        # documentation: OpenGIS simple features polygons consist of one outer ring (linearring), and zero or more inner rings
        #  if not, the function will likely fail.
        xy = shape2imagecoords( geometry, rasterfile_hdr, localprintcommand=localprintcommand )
        xypath = matplotlib.path.Path(xy)
        # xypath.contains_points() function can tell if a point is inside the polygon or not. But we do not want to send the 
        #  whole (potentially huge) raster to be tested. First, subset the raster by the extent of the polygon
        #  make also sure that the indices are within the raster limits
    
        GT = rasterfile.GetGeoTransform() # gives the same information as in "map info"
        # GT: coefficients for transforming between pixel/line (P,L) raster space, and projection coordinates (Xp,Yp) space
        # Xp = GT[0] + P*GT[1] + L*GT[2];
        # Yp = GT[3] + P*GT[4] + L*GT[5];
        # get raster extent. 
        xx = rasterfile.RasterXSize # image size in pixel coords
        yy = rasterfile.RasterYSize 
        bb = xypath.get_extents().corners()
        xmin = max( 0, int( np.floor( bb[0,0] ) ) ) # use upper-left corner
        xmax = min( xx-1, int( np.ceil( bb[3,0] ) ) ) # use lower-right corner
        xrange = xmax - xmin + 1
        
        ymin = max( 0, int( np.floor( bb[0,1] ) ) )# use upper-left corner
        ymax = min( yy-1, int( np.ceil( bb[3,1] ) ) ) # use lower-right corner
        yrange = ymax - ymin + 1
        if xrange > 0 and yrange > 0:
            # the vector is inside the raster
            # next, we need to create an index to slice the input raster
            ii = np.indices( (xrange,yrange) )
            iix = ii[0] # x-coordinates for the indices. Need to be reshaped before usage
            iix = iix.reshape( iix.shape[0]*iix.shape[1] ) + xmin
            iiy = ii[1]
            iiy = iiy.reshape( iiy.shape[0]*iiy.shape[1] ) + ymin
            # a 2-column numpy array with coordinates (huh, that was a lot of work. There has to be an easier solution) XXX
            ii = np.stack( (iix,iiy), 1 )
            polytest = xypath.contains_points( ii ) # this is the actual test.
            j = np.where( polytest )[0]
            pointarray = ii[ j, : ] # extract the points inside the polygon
            
            # output coordlist -- note that x is the second (#1) index; y is the first (#0) in Envi data
            coordlist = [ list( pointarray[:,1] ), list( pointarray[:,0]) ]
        else:
            # return empty lists
            coordlist = [ [], [] ]
    return coordlist
    
def extract_spectrum( hypfilename, pointarray, areasize, areaunit, areashape, hypdata=None, hypdata_map=None ):
    """
    Extract the average spectra for each point in pointarray. The window used in
    extraction depends on input parameters.
    hypfilename: the filename or spectral file  handle of hyperspectral data file
    pointarray: numpy matrix with two columns, in each row x,y coordinates of the points in global coordinates
    areasize: size of the sampled area (circle diameter, square side)
    areaunit: unit in which areasize is given ("meter" or "pixel", first letter suffices)
    areashape: 'circle' or 'square' (first letter suffices)
    hypdata, hypdata_map: spectral handles for file and memmap (optional). If given, hypfilename will not be reopened
    --- outputs
    spectrumlist: list of the extracted spectrum for each point,
    wl: np.array with wavelengths. 
        If no spectral information is available, band numbers with a minus sign are returned
    Nlist: list with the number of pixels averaged for each point
    """
    if hypdata is None:
        # open the file if not open yet. This only gives access to metadata.                
        hypdata = spectral.open_image( hypfilename )
        # open the file as memmap to get the actual hyperspectral data
        hypdata_map = hypdata.open_memmap() # open as BIL

    bands = hypdata_map.shape[2]
    
    wl,use_spectra = get_wavelength( hypfilename, hypdata )

    if 'data ignore value' in hypdata.metadata:
        # actually, Spectral converts DIV's to nan's
        #  so this is unnecessary
        #     but maybe not if the file contains ints?
        use_DIV = True
        dtype = int( hypdata.metadata['data type'] )
        if dtype<4 or dtype > 11:
            DIV = int( float(hypdata.metadata['data ignore value']) )
        else:
            DIV = float( hypdata.metadata['data ignore value'] )
    else:
        DIV = -1 
        use_DIV = False
    
    spectrumlist = []
    Nlist = []
    # loop over points
    for xy_row in pointarray:
        coordlist = pixel_coords( hypfilename, xy_row, areasize, areaunit, areashape, hypdata, hypdata_map )
        if len( coordlist ) > 0:
            spectrum,N = avg_spectrum( hypfilename, coordlist, DIV, hypdata, hypdata_map )
            spectrumlist.append( spectrum )
            Nlist.append( N )
                
            # hypdata_sub = hypdata_map[ coordlist ]
            # if use_DIV:
            #     # look for no data values
            #     hypdata_sub_min = np.min( hypdata_sub, axis=1 )
            #     hypdata_sub_max = np.max( hypdata_sub, axis=1 )
            #     sub_incl = np.where( np.logical_and( hypdata_sub_min!=float('nan'), 
            #         np.logical_or( hypdata_sub_min!=DIV, hypdata_sub_max!=DIV ) ) )[0]
            #     hypdata_sub = hypdata_sub[ sub_incl, : ]
            # spectrumlist.append( np.mean( hypdata_sub, axis=0 ) )
            # Nlist.append( hypdata_sub.shape[0] )
        else:
            spectrumlist.append( [] )
            Nlist.append( 0 )
    return spectrumlist, wl, Nlist
    
import gdal
from osgeo import ogr
from osgeo import osr
import sqlite3
import os

# functions to read data from spatial databases with special focus on geopackage and Metsäkeskus open data
# two broad groups: read the spatial information (geometries) by opening the files as shapefiles (functins vector_*() )
#   and red the data tables in geopackage as sqlite3 databases with no spatial information (functions geopackage_*() )
# + some extras, e.g. a function to subset a list of geometries with a geographic raster

def vector_getfieldnames( filename_in, layernumber=0, localprintcommand=None ):
    """
    get the names of fields in a vector file.
    in: filename, and the number of the layer to get the field names from
    output: a list of field names
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'vector_getfieldnames(): ' # for messaging
    
    # open the file (e.g., geopackage) as shapefile
    sh_file = ogr.Open( filename_in )
    # in geopackage, this only opens the spatial layer
    localprintcommand("File {}: {:d} spatial layer(s).\n".format( filename_in,  sh_file.GetLayerCount() ) )
    fieldnames = [] # the output list
    if sh_file.GetLayerCount() > layernumber:
        sh_layer = sh_file.GetLayerByIndex( layernumber )         
        # get the field names as instructed at 
        # https://gis.stackexchange.com/questions/220844/get-field-names-of-shapefiles-using-gdal
        ldefn = sh_layer.GetLayerDefn()
        for n in range(ldefn.GetFieldCount()):
            fdefn = ldefn.GetFieldDefn(n)
            fieldnames.append(fdefn.name)
    sh_file.Release()
    return fieldnames

def vector_getWKT( infile, layernumber=0, localprintcommand=None):
    """
    returns the projection information in ESRI WKT format
    infile can be a filename or a handle, osgeo.ogr.DataSource
    
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'vector_getWKT(): ' # for messaging

    sh_projectionWkt = None # output
    
    release_infile = False
    if type( infile ) is not ogr.DataSource:
        # open geopackage as shapefile
        infile = ogr.Open( infile )
        release_infile = True
        
    if infile.GetLayerCount() > layernumber:
        sh_layer = infile.GetLayerByIndex( layernumber )         
        sh_SpatialReference = sh_layer.GetSpatialRef()
        # convert to ESRI WKT and then export
        sh_SpatialReference.MorphToESRI()
        sh_projectionWkt = sh_SpatialReference.ExportToWkt()
    if release_infile:
        infile.Release()
    return sh_projectionWkt 

def vector_getSpatialReference( filename_in, layernumber=0, localprintcommand=None):
    """
    returns the projection information, SpatialReference()
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'vector_getWKT(): ' # for messaging

    sh_SpetialReference = None # output
    
    # open geopackage as shapefile
    sh_file = ogr.Open( filename_in )
    if sh_file.GetLayerCount() > layernumber:
        sh_layer = sh_file.GetLayerByIndex( layernumber )         
        sh_SpatialReference = sh_layer.GetSpatialRef().Clone()
        # convert to ESRI WKT and then export
        sh_SpatialReference.MorphToESRI()
    sh_file.Release()
    return sh_SpatialReference
    
def vector_getfeatures( filename_in, fieldnames_in=None, layernumber=0, localprintcommand=None ):
    """
    get the features and the associated spatial entities from a vector file.
    in: filename, 
        the names of the fields to retrieve. if no names are given, just field IDs are retrieved
        the number of the layer to get the field data from
    output: a list with at least two elements:
        1) feature IDs
        2) feature geometries
        if fieldnames_in is not None, followed by lists of field values (one list per field, each list has one element per feature)
         if a specific field does not exist, None is returned as value
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'vector_getfeatures(): ' # for messaging

    # open geopackage or shapefile
    sh_file = ogr.Open( filename_in )
    # in geopackage, this only opens the spatial layer
    # localprintcommand("File {}: {:d} spatial layer(s).\n".format( filename_in,  sh_file.GetLayerCount() ) )
    if sh_file.GetLayerCount() > layernumber:
        sh_layer = sh_file.GetLayerByIndex( layernumber )
        FIDlist = []
        geomlist = []
        # create a list of empty lists to store output values
        valuelist = []
        for fn in fieldnames_in:
            valuelist.append( [] )
        # convert field names to numbers
        ldefn = sh_layer.GetLayerDefn()
        fieldnames = [ ldefn.GetFieldDefn(n).name for n in range(ldefn.GetFieldCount()) ]
        fieldnumbers = [ fieldnames.index(x) if x in fieldnames else -1 for x in fieldnames_in ]
            
        for feature in sh_layer:
            # Lauri saved geometries as strings-- likely he found it to be sufficiently robust, instead of copying the geometries
            # we can get geometry back as follows:
            # geom = ogr.CreateGeometryFromWkt( shapes[key] )
            # however, it makes more sense to save just copies of geometries
            # shapes[str(feature.GetField("standid"))] = str(feature.GetGeometryRef()) 
            geomlist.append( feature.GetGeometryRef().Clone() ) 
            FIDlist.append( feature.GetFID() )
            for vl,fn in zip(valuelist,fieldnumbers):
                vl.append( feature.GetField( fn ) if fn>-1 else None )
        #print(len(FIDlist))
        outlist = [ FIDlist, geomlist ]
        # add the values for requested fields
        for valuesublist in valuelist:
            outlist.append( valuesublist )
    else:
        outlist = []
    sh_file.Release()
    return outlist
    
def vector_createPRJ(filename, wkt):
    """
    Create a projection file for a shapefile from wkt. Wkt string can be created as 
        sh_SpatialReference = sh_layer.GetSpatialRef()
        sh_SpatialReference.MorphToESRI() # not sure if this is 100% necessary
        projectionWkt = sh_SpatialReference.ExportToWkt()
        OR with the vector_getWKT() function in this file
    """
    # force extension to be prj
    filename = os.path.splitext(filename)[0] + ".prj"
    file = open(filename, 'w')
    file.write(wkt)
    file.close()

def vector_swapdata2( infile, featuredict, fieldnames=None, keyfield=None, filename_out=None, layername_out ='', layernumber_in=0, localprintcommand=None ):
    """
    A TEST, DO NOT USE !!!
    create a new shapefile using the spatial data in infile, but with the data table included from featuredict: the keys of featuredict
        are matched to to keyfild to create a spatial conection. In case of no match, no feature is created.
    in: infile: a vector file, either ogr.DataSource or file name to open
        featuredict: a dictionary of field values for features, keys are geometric feature IDs. More than one value can be 
            given for each feature if a list is used as dictionary elements. Lists should be of the same length and have consistent typing for all elements.
        fieldnames: names of the features. If featuredict contains lists, so should featuredict
        keyfield: the field in infile containing the keys to match with featuredict.keys()
            if None or not found in field names, FID is used
        filename_out: the output file name. If not given, a file is created and a handle is returned
        layernumber_in: the number of the layer to get the spatial data from
    output: a file handle (if file is created in memory), or None
    """
        
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'vector_getfeatures(): ' # for messaging
    
    # find out if we have only one field to create, and wrap it in list if necessary
    # Extract and check the first dictionary element for the number of fields
    samplefields = list( featuredict.values() )[0]
    if isinstance( samplefields, str ) or not hasattr( samplefields, '__iter__'):
        # we have one field, wrap it in a list
        for key in featuredict.keys():
            featuredict[ key ] = [ featuredict[ key ] ]
    if fieldnames is not None:
        # make sure fieldnames is also a list
        if isinstance( fieldnames, str ) or not hasattr( fieldnames, '__iter__'):
            fieldnames = [ fieldnames ]    
    # now we can iterate over featuredict values and also fieldnames
    samplefields = list( featuredict.values() )[0] # samplefields has to be a list
    N_fields = len( samplefields )
    
    # just in case, convert featuredict keys to string for robust matching 
    featuredict_str = { str(k):v for k,v in featuredict.items() } 

    if fieldnames is None:
        # use sequential list of numbers as strings for field names.
        fieldnames_range = range( N_fields )
        fieldnames = [ str(i+1) for i in fieldnames_range ]
    
    assert len( fieldnames ) == N_fields 
    
    # open geopackage or shapefile if not open yet
    release_infile = False # if we do not open infile, we shall not close it
    if type( infile ) is not ogr.DataSource:
        # we were given a filename (of a geopackage), open as shapefile
        infile = ogr.Open( infile )
        release_infile = True
    # in geopackage, this only opens the spatial layer
    # find the spatial reference for copying to outfile. There must be a more graceful way (not through wkt)
    infile_wkt = vector_getWKT( infile )
    infile_SpatialReference = osr.SpatialReference()
    infile_SpatialReference.ImportFromWkt( infile_wkt )
    
    all_set = True # a flag that we are ready to go
    
    # create new output file
    release_outfile = False # if a file is created on disk, it will be explicitly closed.
    if filename_out is None:
        outfile = ogr.GetDriverByName('MEMORY').CreateDataSource('memData')
        resulthandle = outfile
    else:
        resulthandle = None # return None if a file is created and closed.
        driver = ogr.GetDriverByName('Esri Shapefile')
        if driver is None:
            localprintcommand( functionname + 'Shape file driver not found, quitting.\n' )
            all_set = False
        else: 
            ds = driver.CreateDataSource( filename_out )
            if ds is None:
                localprintcommand( functionname + 'Could not create output file, quitting.\n' )
                all_set = False
            else:
                vector_createPRJ( filename_out, infile_wkt )
                release_outfile = True
    if all_set:
        outlayer = outfile.CreateLayer( layername_out, srs=infile_SpatialReference, geom_type=ogr.wkbPolygon )
    
        # create new fields in outfile 
        outlayerDefn = outlayer.GetLayerDefn()
        for fieldname,value in zip(fieldnames,samplefields):
            fieldDefn = ogr.FieldDefn(fieldname, OGR_FIELD_TYPES_MAP[ type( value ) ] )
            outlayer.CreateField(fieldDefn)
        
        if infile.GetLayerCount() > layernumber:
            inlayer = infile.GetLayerByIndex( layernumber )         

            # find the field number to match with featuredict_str.keys()
            ldefn = inlayer.GetLayerDefn()
            keyfieldnumber = fieldnames.index( keyfieldname ) if x in fieldnames else -1
            
            # loop over geographic features
            for feature in inlayer:
                idtomatch = feature.GetField( keyfieldnumber ) if keyfieldnumber>-1 else feature.GetFID()
                # convert to string for robust matching 
                idtomatch = str( idtomatch )
                
                if idtomatch in featuredict_str.keys():
                    # Create feature in output file
                    feat = ogr.Feature( outlayerDefn )
                    feat.SetGeometry( feature.GetGeometryRef().Clone() )
                    for fieldname in fieldnames:
                        feat.SetField( fieldname, featuredict_str[ idtomatch ] )
                    
        else:
            localprintcommand( functionname + 'Could not find layer {}, quitting.\n'.format( str( layernumber )) )
            
    # close output and input file if required
    if release_outfile:
        outfile.Release()
    if release_infile:
        infile.Release()
    return resulthandle

def vector_newfile( geometrylist, featuredict, filename_out=None, layername_out ='', localprintcommand=None ):
    """
    create a new shapefile using the spatial data in geometrylist and the data table in featuredict
        the keys of featuredict are field names, dictionary elements should be lists of equal length, and the same length as geometrylist
    in: 
        geometrylist: a  list of geometries (OGRSpatialReference), currently have to be ogr.wkbPolygon
        featuredict: a dictionary of field values for features, keys are field names, values are lists of the same length as geometrylist. 
            All lists should be of the same length and have consistent typing for all elements.
        filename_out: the output file name. If not given, a file is created and a handle is returned
    output: a file handle (if file is creted in memory), or None
    """
        
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'vector_getfeatures(): ' # for messaging

    # Make sure featuredict keys are strings
    featuredict_str = { str(k):v for k,v in featuredict.items() } 
    fieldnames = list( featuredict_str.keys() )

    # find out if we have only one feature to create, and wrap it in list if necessary
    # Extract and check the first dictionary element for the number of fields
    samplefields = list( featuredict.values() )[0]
    if isinstance( samplefields, str ) or not hasattr( samplefields, '__iter__'):
        # we have one field, wrap it in a list
        for key in featuredict.keys():
            featuredict[ key ] = [ featuredict[ key ] ]
    if not hasattr( geometrylist, '__iter__'):
        geometrylist = [ geometrylist ]
    # now we can iterate over featuredict values and also geometrylist
    N_features= len( samplefields )
    
    # create sample fields for defining data table types
    # take the first element of all elements in featuredict
    samplefields = [ featuredict_str[x][0] for x in fieldnames ]
    
    assert len( geometrylist ) == N_features
    
    # get spatial reference from the first element in geometrylist
    out_SpatialRef = geometrylist[0].GetSpatialReference()
    
    all_set = True # a flag that we are ready to go
    
    # create new output file
    release_outfile = False # if a file is created on disk, it will be explicitly closed.
    if filename_out is None:
        outfile = ogr.GetDriverByName('MEMORY').CreateDataSource('memData')
        resulthandle = outfile
    else:
        resulthandle = None # return None if a file is created and closed.
        driver = ogr.GetDriverByName('Esri Shapefile')
        if driver is None:
            localprintcommand( functionname + 'Shape file driver not found, quitting.\n' )
            all_set = False
        else: 
            ds = driver.CreateDataSource( filename_out )
            if ds is None:
                localprintcommand( functionname + 'Could not create output file, quitting.\n' )
                all_set = False
            else:
                vector_createPRJ( filename_out, infile_wkt )
                release_outfile = True
    # search for spatial reference in 
    outSpatialRef = None
    for geom_i in geometrylist:
        outSpatialRef = geom_i.GetSpatialReference()
        if outSpatialRef is not None:
            break
    
    if all_set:

        outlayer = outfile.CreateLayer( layername_out, srs=outSpatialRef, geom_type=ogr.wkbPolygon )
    
        # create new fields in outfile 
        outlayerDefn = outlayer.GetLayerDefn()
        for fieldname,value in zip(fieldnames,samplefields):
            fieldDefn = ogr.FieldDefn(fieldname, OGR_FIELD_TYPES_MAP[type(value)] )
            outlayer.CreateField(fieldDefn)

        # loop over geographic features
        for i,geometry in enumerate(geometrylist):
            # Create feature in output file
            feat = ogr.Feature( outlayerDefn )
            feat.SetGeometry( geometry.Clone() )
            for fieldname in fieldnames:
                feat.SetField( fieldname, featuredict_str[ fieldname ][i] )
            outlayer.CreateFeature( feat )

    else:
        localprintcommand( functionname + "could not create the shapefile" )
        
    # close output file if required
    if release_outfile:
        outfile.Release()
    return resulthandle
    
def vector_rasterize_like( shpfile, rasterfile, shpfield=None, layernumber=0, dtype=None, RasterizeOptions=[], DIV=0, localprintcommand=None ):
    """
    Based on rasterize_shapefile_like() from project rastercube   Author: terrai   File: shputils.py
        https://github.com/terrai/rastercube, MIT License (checked 22.5.2018)
    Given a shapefile, rasterizes it in memory so it has the exact same extent as the rasterfile.
    in:
        shpfile: file to rasterize
        rasterfile: the sample raster used to get geometry from
        shpfield: (str) name of the field in shapefile to get the raster levels from. If none, a constant value of one is used to create a mask
        layernumber: which layer is shpfile to use
        dtype: pyhton data type to for raster. If None, determined based on the values in shpfield
            dtype is converted to gdal type using GDAL_FIELD_TYPES_MAP defined at the beginning of this file
        RasterizeOptions: a list that will be passed to GDAL RasterizeLayers -- papszOptions, like ["ALL_TOUCHED=TRUE"]
            shpfield is added to the beginning of the options sent to RasterizeLayers 
        DIV: data ignore value (no data value) which is used to initiate the raster.
    out:
        numpy array (via ReadAsArray() )
    """
            
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'vector_rasterize_like(): ' # for messaging
    
    if type( rasterfile ) is gdal.Dataset:
        model_dataset = rasterfile
    else:
        # rasterfile_ is either envi hdr or envi datafile name
        rasterfile_data, rasterfile_hdr = envifilecomponents( rasterfile, localprintcommand=localprintcommand )
        model_dataset = gdal.Open( rasterfile_data )
        
    if type( shpfile ) is ogr.DataSource:
        shape_name = shpfile.GetDescription()
        shape_dataset = shpfile
    else:
        # we were given a filename (of a geopackage), open as shapefile
        shape_name = shpfile
        shape_dataset = ogr.Open( shpfile )
        
    if shape_dataset.GetLayerCount() > layernumber:
        shape_layer = shape_dataset.GetLayerByIndex( layernumber )
        if dtype is None:
            if shpfield is not None:
                #get the first value from the shpfile, shpfield
                dtype = type ( shape_layer.GetNextFeature().GetField( shpfield ) )
                shape_layer.ResetReading() # rewind, just in case
            else:
                # we were asked to create a mask
                dtype = int
                
        # convert dtype to GDAL type
        dtype_GDAL = GDAL_FIELD_TYPES_MAP[ dtype ]
        
        mem_drv = gdal.GetDriverByName('MEM')
        mem_raster = mem_drv.Create( '', model_dataset.RasterXSize, model_dataset.RasterYSize, 1, dtype_GDAL )
        mem_raster.SetProjection(model_dataset.GetProjection())
        mem_raster.SetGeoTransform(model_dataset.GetGeoTransform())
        mem_band = mem_raster.GetRasterBand(1)
        mem_band.Fill(DIV)
        mem_band.SetNoDataValue(DIV)
    
        # http://gdal.org/gdal__alg_8h.html#adfe5e5d287d6c184aab03acbfa567cb1
        # http://gis.stackexchange.com/questions/31568/gdal-rasterizelayer-doesnt-burn-all-polygons-to-raster
        if shpfield is None:
            # create  a mask
            burn_value = [ ( dtype )( 1.0 ) ] # create a burn value of the correct data type (although likely unnecessary)
            err = gdal.RasterizeLayer( mem_raster, [1], shape_layer, burn_values=[burn_value], options=RasterizeOptions )
        else:
            RasterizeOptions.insert( 0, "ATTRIBUTE="+shpfield ) # add the field to use in the beginning of option list 
            err = gdal.RasterizeLayer( mem_raster, [1], shape_layer, options=RasterizeOptions )

        assert err == gdal.CE_None
        return mem_raster.ReadAsArray() 
    else:
        localprintcommand( functionname + "could not load layer #{:d} from file {}."
            .format( layernumber, shape_name ) )
        return None

def vector_rasterize( shpfile, rasterfile, shpfield=None, layernumber=0, band=1, RasterizeOptions=[], localprintcommand=None ):
    """
    Rasterize a shapefile into an existing raster shapefile, rasterizes it in memory so it has the exact same extent as the rasterfile.
    NOTE: seems to give error "ERROR 3: Failed to write scanline 0 to file" when doing raster.FlushCache() on an ENVI file
    in:
        shpfile: file to rasterize
        rasterfile: the existing file which will be written into
        shpfield: (str) name of the field in shapefile to get the raster levels from. If none, a constant value of one is used to create a mask
        layernumber: which layer is shpfile to use
        band: band number in rasterfile to modify -- this band will include the rastrized vector
        RasterizeOptions: a list that will be passed to GDAL RasterizeLayers -- papszOptions, like ["ALL_TOUCHED=TRUE"]
            shpfield is added to the beginning of the options sent to RasterizeLayers 
    out:
        True or False (success or failure)
    """
            
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'vector_rasterize(): ' # for messaging
    
    if type( rasterfile ) is gdal.Dataset:
        raster = rasterfile
    else:
        # rasterfile_ is either envi hdr or envi datafile name
        rasterfile_data, rasterfile_hdr = envifilecomponents( rasterfile, localprintcommand=localprintcommand )
        raster = gdal.Open( rasterfile_data )
        
    if type( shpfile ) is ogr.DataSource:
        shape_name = shpfile.GetDescription()
        shape_dataset = shpfile
    else:
        # we were given a filename (of a geopackage), open as shapefile
        shape_name = shpfile
        shape_dataset = ogr.Open( shpfile )
        
    if shape_dataset.GetLayerCount() > layernumber:
        shape_layer = shape_dataset.GetLayerByIndex( layernumber )
        
        if band <= raster.RasterCount:
            
            if shpfield is None:
                # create  a mask
                burn_value = 1 
                err = gdal.RasterizeLayer( raster, [ band ], shape_layer, burn_values=[burn_value], options=RasterizeOptions )
            else:
                RasterizeOptions.insert( 0, "ATTRIBUTE="+shpfield ) # add the field to use in the beginning of option list 
                err = gdal.RasterizeLayer( raster, [band], shape_layer, options=RasterizeOptions )
    
            assert err == gdal.CE_None
            raster.FlushCache()
            return True
        else:
            localprintcommand( functionname + "could not load band #{:d} from raster file {}."
                .format( band, rasterfile) )
    else:
        localprintcommand( functionname + "could not load layer #{:d} from file {}."
            .format( layernumber, shape_name ) )
        return False
    
    
def geopackage_getdatatables( datasource, localprintcommand=None ):
    """
    get the data tables from a geopackage file.
    in: datasource -- either filename, sqlite2.Connection or Sqlite3.Cursor
    out: a list with table names
    """
    
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'geopackage_getdatatables(): ' # for messaging
    
    # open geopackage as sqlite database if necessary
    if type(datasource) == str:
        conn = sqlite3.connect( datasource ) 
        closefile = True
    else: 
        conn = datasource
        closefile = False
    # conn is now either sqlite3.Connection or sqlite3.Cursor
    c = conn.cursor() if type( conn ) == sqlite3.Connection else conn
    
    tablelist = []
    res = c.execute("SELECT name FROM sqlite_master WHERE type='table';")

    for tablename in res:
        tablelist.append( tablename[0] )
    if closefile:
        conn.close()
    return tablelist
    
def geopackage_getfieldnames( datasource, tablename, localprintcommand = None ):
    """
    get the field names in a data table of a geopackage (sqlite3) file.
    in: datasource -- either filename, sqlite2.Connection or Sqlite3.Cursor
        tablename: (str) name of the table to list
    out: a list with field names
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'geopackage_getfieldnames(): ' # for messaging
    
    # open geopackage as sqlite database if necessary
    if type(datasource) == str:
        conn = sqlite3.connect( datasource ) 
        closefile = True
    else: 
        conn = datasource
        closefile = False
    # conn is now either sqlite3.Connection or sqlite3.Cursor
    c = conn.cursor() if type( conn ) == sqlite3.Connection else conn
   
    tablelist = geopackage_getdatatables( c, localprintcommand=localprintcommand )
    fieldnames = []
    if tablename in tablelist:
        res = c.execute ("PRAGMA table_info({});".format(tablename) )
        for fi in res:
            fifn = fi[1]
            fieldnames.append(fifn)
            
    if closefile:
        conn.close()
    return fieldnames

def geopackage_getuniquevalues( datasource, tablename, fieldnames, additionalconstraint='', localprintcommand=None ):
    """
    get the unique values in a field in a data table of a geopackage (sqlite3) file.
    in: datasource -- either filename, sqlite2.Connection or sqlite3.Cursor
        tablename: name of the table in which the filed is
        fieldnames: (list of) name(s) of the field for which the values will be returned
        additionalconstraint: (optional) a string to add to the sql query, e.g. " where field2 = 34 "
    out: a list with values; if more than one value was requested, a list of lists is returned
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'geopackage_getuniquevalues(): ' # for messaging
    
    # open geopackage as sqlite database if necessary
    if type(datasource) == str:
        conn = sqlite3.connect( datasource ) 
        closefile = True
    else: 
        conn = datasource
        closefile = False
    # conn is now either sqlite3.Connection or sqlite3.Cursor
    c = conn.cursor() if type( conn ) == sqlite3.Connection else conn

    # make sure we can loop over fieldnames -- python thinks strings should be iterated over
    if isinstance(fieldnames, str):
        fieldnames = [ fieldnames ]
        unpack = True # unpack the list of lists before returning
    else:
        unpack = False

    uniquevalues = []
    for fieldname in fieldnames:
        res = c.execute("select distinct {0} from {1} {2} order by {0};"
            .format( fieldname, tablename, additionalconstraint ) )
        uniquevalues_i = []
        for row in res:
            uniquevalues_i.append(row[0])
        uniquevalues.append( uniquevalues_i )
    if unpack:
        uniquevalues = uniquevalues[0]
    return uniquevalues
    
def geopackage_getvalues( datasource, tablename, fieldnames, additionalconstraint='', localprintcommand=None ):
    """
    get the values in a field in a data table of a geopackage (sqlite3) file.
    in: datasource -- either filename, sqlite2.Connection or Sqlite3.Cursor
        tablename: name of the table in which the filed is
        fieldnames: (list of) name(s) of the field for which the values will be returned
        additionalconstraint: (optional) a string to add to the sql query, e.g. " where field2 = 34 "
    out: a list with values; if more than one value was requested, a list of lists is returned
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'geopackage_getvalues(): ' # for messaging
    
    # open geopackage as sqlite database if necessary
    if type(datasource) == str:
        conn = sqlite3.connect( datasource ) 
        closefile = True
    else: 
        conn = datasource
        closefile = False
    # conn is now either sqlite3.Connection or sqlite3.Cursor
    c = conn.cursor() if type( conn ) == sqlite3.Connection else conn
    
    # make sure we can loop over fieldnames -- python thinks strings should be iterated over
    if isinstance(fieldnames, str):
        fieldnames = [ fieldnames ]
        unpack = True # unpack the list of lists before returning
    else:
        unpack = False
    
    values = []
    for fieldname in fieldnames:
        # go thorugh the requested fields one by one
        res = c.execute("select {0} from {1} {2} ;"
            .format( fieldname, tablename, additionalconstraint ) )
        values_i = []
        for row in res:
            values_i.append(row[0])
        values.append( values_i )
    if unpack:
        values = values[0]
    return values

def geopackage_getspecificvalues1( datasource, tablename, fieldname, values_in, fieldname_in, additionalconstraint='', localprintcommand=None ):
    """
    get the values in a field in a data table of a geopackage (sqlite3) file for specific data rows
    the specific data rows are determined by the MANY VALUES in value_in, searched for in fieldname_in
    in: datasource -- either filename, sqlite3.Connection or sqlite3.Cursor
        tablename: name of the table in which the field is
        fieldname: name of the field for which the values will be returned
        values_in: (list of) the value(s) to search for in fieldname_in
        fieldname_in: the field to search for value_in
            fieldname_in == value_in[i] is matched for each output value
        additionalconstraint: (optional) a string to add to the sql query, e.g. " where field2 = 34 "
    out: a list with values; if more than one value was requested, a list of lists is returned
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'geopackage_getspecificvalues(): ' # for messaging
    
    # open geopackage as sqlite database if necessary
    if type(datasource) == str:
        conn = sqlite3.connect( datasource ) 
        closefile = True
    else: 
        conn = datasource
        closefile = False
    # conn is now either sqlite3.Connection or sqlite3.Cursor
    c = conn.cursor() if type( conn ) == sqlite3.Connection else conn
    
    # make sure we can loop over fieldnames -- python thinks strings should be iterated over
    if isinstance(values_in, str):
        values_in = [ values_in ]
        unpack = True # unpack the list of lists before returning
    else:
        unpack = False
    
    values = []
    for value in values_in:
        # go through the requested values one by one
        res = c.execute("select {0} from {1} where {2}={3} {4};"
            .format( fieldname, tablename, fieldname_in, value, additionalconstraint ) )
        values.append( c.fetchall() )
        
    if closefile:
        conn.close()
        
    if unpack:
        values = values[0]
    return values

def geopackage_getspecificvalues2( datasource, tablename, fieldnames, value_in, fieldname_in, additionalconstraint='', localprintcommand=None ):
    """
    get the values in a field in a data table of a geopackage (sqlite3) file for specific data rows
    the specific data rows are determined by the SINGLE value in value_in, searched for in fieldname_in
    in: datasource -- either filename, sqlite3.Connection or Sqlite3.Cursor
        tablename: name of the table in which the field is
        fieldnames: (a list of) name(s) of the field(s) for which the values will be returned
        value_in: (str) the value to search for in fieldname_in 
        fieldname_in: the field to search for value_in
        additionalconstraint: (optional) a string to add to the sql query, e.g. " where field2 = 34 "
    out: a list with values; if values from more than one field was requested, a list of lists is returned
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'geopackage_getspecificvalues2(): ' # for messaging
    
    # open geopackage as sqlite database if necessary
    if type(datasource) == str:
        conn = sqlite3.connect( datasource ) 
        closefile = True
    else: 
        conn = datasource
        closefile = False
    # conn is now either sqlite3.Connection or sqlite3.Cursor
    c = conn.cursor() if type( conn ) == sqlite3.Connection else conn
    
    
    values = [] # output variable
    
    # merge the requested field values into a comma-separated string
    if isinstance(fieldnames, str):
        fieldsasstring = fieldnames
    else:
        fieldsasstring = ",".join( fieldnames ) 
        
    # do the query and fetch results
    res = c.execute("select {0} from {1} where {2}={3} {4};"
        .format( fieldsasstring, tablename, fieldname_in, value_in, additionalconstraint ) )
    outcome = c.fetchall()
    # rearrange the output list, now it's grouped in outcome by record (row). Regroup by field (column)
    if len( outcome ) > 0:
        for i in range( len(outcome[0] ) ):
            values.append( [ value[i] for value in outcome ] )
    else:
        localprintcommand(functionname+" no data with matching criteria found.\n")
        
    if closefile:
        conn.close()
        
    # if we only have one field value requested, return list, not list of lists
    if isinstance(fieldnames, str):
        values = values[0]
    return values
    
def geopackage_uniquevalues( datasource, tablename, fieldname, additionalconstraint='', localprintcommand=None ):
    """
    get all unique values in a field in a data table of a geopackage (sqlite3) file 
    in: datasource -- either filename, sqlite2.Connection or Sqlite3.Cursor
        tablename: name of the table in which the filed is
        fieldname: name of the field for which the unique values will be returned
        additionalconstraint: (optional) a string to add to the sql query, e.g. " where field2 = 34 "
    out: a dictionary with value:number pairs
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'geopackage_uniquevalues(): ' # for messaging
    
    # open geopackage as sqlite database if necessary
    if type(datasource) == str:
        conn = sqlite3.connect( datasource ) 
        closefile = True
    else: 
        conn = datasource
        closefile = False
    # conn is now either sqlite3.Connection or sqlite3.Cursor
    c = conn.cursor() if type( conn ) == sqlite3.Connection else conn
    
    res = c.execute("select {0} from {1} {2};"
        .format( fieldname, tablename, additionalconstraint ) )
    outcome = c.fetchall()
    uniquevalues = {}
    for row in outcome:
        if row[0] in uniquevalues.keys():
            uniquevalues[ row[0] ] += 1
        else:
            uniquevalues[ row[0] ] = 1
    return uniquevalues

def geopackage_countvalues( datasource, tablename, fieldname, fieldname_in, additionalconstraint='', localprintcommand=None ):
    """
    count the number of values and unique values in a field in a data table of a geopackage (sqlite3) file 
        for each unique value in another field, fieldname_in. Returns only the summary of counts, not the values for each unige value in fieldname_in
    in: datasource -- either filename, sqlite2.Connection or Sqlite3.Cursor
        tablename: name of the table in which the filed is
        fieldname: name of the field for which the counts will be returned
        fieldname_in: the field to search for value_in
        additionalconstraint: (optional) a string to add to the sql query, e.g. " where field2 = 34 "
    out: a dictionary with "count,uniquecount":number pairs
        "count,uniquecount" are the combinations of values as strings
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'geopackage_countvalues(): ' # for messaging
    
    # open geopackage as sqlite database if necessary
    if type(datasource) == str:
        conn = sqlite3.connect( datasource ) 
        closefile = True
    else: 
        conn = datasource
        closefile = False
    # conn is now either sqlite3.Connection or sqlite3.Cursor
    c = conn.cursor() if type( conn ) == sqlite3.Connection else conn
    
    # get unique values in fieldname_in
    uniquevalues = geopackage_uniquevalues( datasource, tablename, fieldname_in, localprintcommand=localprintcommand )
    
    valuedict = {}
    # loop over the retrieved unique values and perform searches in the database
    for value_in in uniquevalues:        
        res = c.execute("select {0} from {1} where {2}={3} {4};"
            .format( fieldname, tablename, fieldname_in, value_in, additionalconstraint ) )
        outcome = c.fetchall()
        # outcome is packed in tuples, but this should not hinder counting and finding unique values
        n_values = len( outcome )
        n_unique = len( set(outcome) )
        stringkey = ",".join( ( str(n_values) , str(n_unique) ) )
        if stringkey in valuedict.keys():
            valuedict[ stringkey ] += 1
        else:
            valuedict[ stringkey ] = 1
    return valuedict
    
def geometries_subsetbyraster( geometrylist, rasterfile_in, reproject=True, localprintcommand=None ):
    """
    give an index of geometries which are (potentially, based on extents) inside the area of rasterfile
    The selection is based on x,y rectangles, so it's not absolutely accurate and may include 
        some geometries just outside the raster area (close to its corners)
    will reproject geometries to raster coordinate system  if reproject is True -- can be slow
    in:  a  list of geometries (OGRSpatialReference)
        rasterfile_in (envi) filename
        reproject: whether to reproject the geometries. 
            If raster file  and the vector shapes are known to be in the same system, reprojection is not necessary
    out:
        a list of integers
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'geometries_subsetbyraster(): ' # for messaging
    
    # open the raster file with gdal to get geometry
    # for envi files: gdal wants the name of the data file, not hdr
    rasterfile = envihdr2datafile( rasterfile_in )
    if rasterfile == '' :
        localprintcommand(functionname + "Cannot find the data file corresponding to {}, ".format(rasterfile_orig) )

    f1_gdal = gdal.Open( rasterfile )
    outlist = []
    if f1_gdal is None:
        localprintcommand( functionname + "cannot open file {}.\n".format( rasterfile ) )
    else:
        f1_wkt = f1_gdal.GetProjection()
        f1_SpatialReference = osr.SpatialReference()
        f1_SpatialReference.ImportFromWkt(f1_wkt)
        localprintcommand( functionname + "Raster has geometry {}\n".format(f1_SpatialReference.ExportToProj4()) )
        
        rasterextent = get_rasterextent_gdal( f1_gdal )
    
        #go through the geometries
        for i,geom in enumerate(geometrylist):
            if reproject:
                geom_temp = geom.Clone() # make a copy not to modify the original
                # make no assumptions -- geometries can have different projections
                vr_transform = osr.CoordinateTransformation( geom_temp.GetSpatialReference(), f1_SpatialReference )
                geom_temp.Transform( vr_transform )
            else:
                geom_temp = geom
            # get spatial extent
            extent = geom.GetEnvelope()
            
            # test the extents
            if extent[1] > rasterextent[0] and extent[0] < rasterextent[1] and extent[3] > rasterextent[2] and extent[2] < rasterextent[3]:
                outlist.append(i)
    return outlist

def get_rasterextent_gdal( rasterfile ):
    """
    Get the extent of a raster in projected coordinates
    in: osgeo.gdal.Dataset or a filename
    out: a list of [ xmin, xmax, ymin, ymax ]
    """
    if type( rasterfile ) is not gdal.Dataset:
        rasterfile = gdal.Open( rasterfile )
    
    GT = rasterfile.GetGeoTransform() # gives the same information as in "map info"
    # GT: coefficients for transforming between pixel/line (P,L) raster space, and projection coordinates (Xp,Yp) space
    # Xp = GT[0] + P*GT[1] + L*GT[2];
    # Yp = GT[3] + P*GT[4] + L*GT[5];
    # get raster extent. project corners
    xx = rasterfile.RasterXSize # max x in pixel coords
    yy = rasterfile.RasterYSize # max y in pixel coords
    # calculate the X,Y of image corners
    X00 = GT[0]
    X01 = GT[0]            + yy*GT[2]
    X10 = GT[0] + xx*GT[1] 
    X11 = GT[0] + xx*GT[1] + yy*GT[2]
    Y00 = GT[3] 
    Y01 = GT[3]            + yy*GT[5]
    Y10 = GT[3] + xx*GT[4] 
    Y11 = GT[3] + xx*GT[4] + yy*GT[5]
    xmin = min( [ X00, X01, X10, X11 ] )
    xmax = max( [ X00, X01, X10, X11 ] )
    ymin = min( [ Y00, Y01, Y10, Y11 ] )
    ymax = max( [ Y00, Y01, Y10, Y11 ] )
    return [ xmin, xmax, ymin, ymax ]
    
def envihdr2datafile( hdrfilename, localprintcommand=None ):
    """
    try to locate the data file associated with the ENVI header file hdrfilename
    because gdal wants the name of the data file, not hdr
    out:
        the full filename of the data file
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'envihdr2datafile(): ' # for messaging
    
    # for envi files: gdal wants the name of the data file, not hdr
    hdrfilename_split = os.path.splitext( hdrfilename )
    
    if hdrfilename_split[1] == ".hdr":
        datafilename = hdrfilename_split[0]
        if not os.path.exists(datafilename):
            # try different extensions, .dat and .bin and .bil
            basefilename = datafilename
            datafilename += '.dat'
            if not os.path.exists(datafilename):
                datafilename  = basefilename + '.bin'
                if not os.path.exists(datafilename):
                    datafilename  = basefilename + '.bil'
                    if not os.path.exists(datafilename):
                        localprintcommand(functionname + "Cannot find the data file corresponding to {}.\n".format(hdrfilename) )
                        datafilename = ''
    return datafilename
    
def envidata2hdrfile( envidatafilename, localprintcommand=None ):
    """
    try to locate the header file associated with the ENVI data file envidatafilename
    because gdal wants the name of the data file, not hdr; but sometimes we need hdr.
    out:
        the full filename of the header file
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'datafile2envihdr(): ' # for messaging
    
    # for envi files: gdal wants the name of the data file, not hdr
    basefilename = os.path.splitext( envidatafilename )[0]
    hdrfilename = basefilename + '.hdr'
    if not os.path.exists(hdrfilename):
        # try just adding hdr to datafile 
        hdrfilename = envidatafilename + '.hdr'
        if not os.path.exists(hdrfilename):
            # no idea how to proceed
            hdrfilename = ''
            localprintcommand(functionname + "Cannot find the hdr file corresponding to {}.\n".format(envidatafilename) )
    return hdrfilename

def envifilecomponents( filename_in, localprintcommand=None ):
    """
    Tries to guess the envi data file and header file names from filename_in
    filename_in is either data or header file
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'envifilecomponents(): ' # for messaging
    
    base_in, extension_in = os.path.splitext( filename_in)
    if  extension_in == ".hdr" or extension_in == ".HDR":
        headerfile = filename_in
        datafile = envihdr2datafile( headerfile, localprintcommand=localprintcommand  )
    else:
        # assume we were given the data file name
        datafile = filename_in
        headerfile = envidata2hdrfile( datafile, localprintcommand=localprintcommand )
    return datafile, headerfile

def envi_addheaderfield( envifilename, fieldname, values, checkifexists=True, localprintcommand=None ):
    """
    Adds a aline to ENVI header file. This function is in gdal-functions because it depends on envifilecomponents.
    ENVI file should be closed before rewriting.
    envifilename: string, file name
    fieldname: name of the field to add
    values: the value to add. Can be a list, e.g. one per band
    checkifexists: flag -- whether to stop if the field already exists
    """
    
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'envi_addheaderfield(): ' # for messaging

    datafile,hdrfile = envifilecomponents( envifilename, localprintcommand=localprintcommand )
    
    if checkifexists and fieldname in open(hdrfile).read() :
        localprintcommand( functionname +" field <{}> already exists in {}. Stopping.\n"
            .format( fieldname, hdrfile ) )
    else:
        with open(hdrfile,'a') as hfile:
            valuestr = [ str(i) for i in values ]
            outstr = fieldname + " = {" + ", ".join(valuestr) + "}"
            hfile.write( outstr )
        localprintcommand( functionname +" Added field <{}> to {}.\n"
            .format( fieldname, hdrfile ) )