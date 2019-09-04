"""
miscellaneous helping functions for hypdatatools
"""

import numpy as np
import pickle

hyperspectral_datafolder = '.'
spectralsensitivity_folder = '.' 
datafolders_loaded = False #whether an attempt has been made to read datafolders

def readtextfile( filename, converttonumbers=True, localprintcommand=None ):
    # try to read a csv file with a varying number of header lines 
    # tries a number of different separators
    # if converttonumbers==True, tries for non-numeric rows at the top of the file.
    # returns a ndarray of float and a ndarray column headings as strings (can be one or more rows)
    # converrtoumbers: if True, attempt conversion to float and searches for a header line with column names
    #    if False, return an np.array of unicode strings
    # localprintcommand: the command used for messaging
    
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'readtextfile(): ' # for messaging
    
    readxy=False 
    # try different separators numbers of header rows 
    # read as unicode text, so maybe it's all unnecessary? -- but there may be commentary lines with 
    for rowstoskip in range(3):
        for sep in (';',',','\t',' '):
            try:
                xy = np.loadtxt( filename, skiprows=rowstoskip, delimiter=sep, dtype='unicode' )
            except ValueError:
                pass
            else:
                if len( xy.shape ) > 1 :
                    # stop only if more than one column is retrieved in xy
                    localprintcommand( functionname + filename + ": using separator [" + sep + "], skipping %i lines.\n" % rowstoskip )
                    readxy=True
                    break
        if readxy:
            break
    
    headings = []
    if readxy and converttonumbers:
        # attempt converting to numbers and separating column titles
        converted = False
        maxrows = min( 3, xy.shape[0]-1 ) # maximum number of header rows allowed
        for xyrowstoskip in range(maxrows,-1,-1):
            try:
                xy_float = xy[xyrowstoskip:,:].astype(float)
            except ValueError:
                break # conversion unsuccessful -- this means we have included a non-number row = heading row
            else:
                converted=True # it's possible to convert input into a float matrix
        if converted:
            # this means that the xy contains floats 
            if xyrowstoskip > -1:
                # additionally, xy contains at least one header row
                headings = xy[ 0:(xyrowstoskip),: ]
                localprintcommand( functionname + "found {:d} header rows.\n".format(xyrowstoskip) )
            return xy_float,headings
        else:
            localprintcommand( functionname + "conversion to floats failed.\n")
    if not readxy:
        localprintcommand( functionname + "could not read the structure in {}.\n".format(filename) )
    # return the unconverted xy (as strings), and empty headings
    return xy, headings

def savehyperspectralfolders( localprintcommand=None ):
    """
    save the default folders hyperspectral folders in the file hyperspectral_datafolders.txt
    saves it in the current folder
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'loadhyperspectralfolders(): ' # for messaging
    
    global hyperspectral_datafolder
    global spectralsensitivity_folder
        
    hypfolderdict = { 'hyperspectral_datafolder' : hyperspectral_datafolder,
        'spectralsensitivity_folder' : spectralsensitivity_folder  }
    
    with open('hyperspectral_datafolders.txt','wb') as hypf:
        pickle.dump( hypfolderdict, hypf )

def loadhyperspectralfolders( localprintcommand=None ):
    """
    attempts to load the default folders hyperspectral folders in the file hyperspectral_datafolders.txt
    saves it in the current folder
    """
    if localprintcommand is None:
        # use a print command with no line feed in the end. The line feeds are given manually when needed.
        localprintcommand = lambda x: print(x,end='')
    functionname = 'loadhyperspectralfolders(): ' # for messaging

    global hyperspectral_datafolder
    global spectralsensitivity_folder 
    global datafolders_loaded 
    
    try:
        with open('hyperspectral_datafolders.txt','rb') as hypf:
            hypfolderdict = pickle.load( hypf )
    except FileNotFoundError:
        localprintcommand(functionname + " could not find file to load default folders.\n")
        hypfolderdict = { 'hyperspectral_datafolder' : hyperspectral_datafolder,
            'spectralsensitivity_folder' : spectralsensitivity_folder }
        
    if 'hyperspectral_datafolder' in hypfolderdict:
        hyperspectral_datafolder = hypfolderdict['hyperspectral_datafolder']
        
    if 'spectralsensitivity_folder' in hypfolderdict:
        spectralsensitivity_folder = hypfolderdict['spectralsensitivity_folder']
    datafolders_loaded = True
    
def get_hyperspectral_datafolder( localprintcommand=None ):
    global hyperspectral_datafolder
    global datafolders_loaded 
    if not datafolders_loaded:
        loadhyperspectralfolders( localprintcommand=localprintcommand )
    return hyperspectral_datafolder
    
def get_spectralsensitivity_folder( localprintcommand=None ):
    global spectralsensitivity_folder 
    global datafolders_loaded 
    if not datafolders_loaded:
        loadhyperspectralfolders( localprintcommand=localprintcommand )
    return spectralsensitivity_folder

def set_hyperspectral_datafolder( h , localprintcommand=None ):
    global hyperspectral_datafolder
    hyperspectral_datafolder = h
    savehyperspectralfolders( localprintcommand=localprintcommand )

def set_spectralsensitivity_folder ( h , localprintcommand=None ):
    global spectralsensitivity_folder 
    spectralsensitivity_folder = h
    savehyperspectralfolders( localprintcommand=localprintcommand )

