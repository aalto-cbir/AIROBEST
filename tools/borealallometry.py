# Allometric functions for pine, spruce and birch for Finland
import numpy as np

# missing data using biometry, mostly based on Majasalmi 2013, with exception of crown radius

def dlw( treespecies, treeheight, trunkdiameter ):
    # dry leaf weight, kg per tree
    #  in: tree height in m, tree trunk diameter (dbh) in cm.
    # Repola (2008,2009), models based on inventory data only: dbh and height
    # pine: Repola 2009 model 1a, Eq. (7)
    # spruce: Repola 2009 model 1c, Eq. (15)
    # birch: Repola 2008 separate model I, Eq. (12)
    #   for dbh<13.7 cm, Johansson's (1999) model as given by Majasalmi et al. 2013, Eq. (8)
    h = treeheight
    dS = 2+1.25*trunkdiameter
    if treespecies == 1:
        # pine
        dlw_i = np.exp( -6.303 + 14.472*dS/(dS+6) -3.976*h/(h+1)  + (0.109+0.118)/2 ) 
    elif treespecies == 2:
        # spruce
        dlw_i = np.exp( -2.994 + 12.251*dS/(dS+10) -3.415*h/(h+1)  +  (0.107+0.089)/2 )
    else:
        # birch or other broadleaf
        if trunkdiameter_i > 11:
            # Repola (2008) had only large trees in the sample
            dlw_i = np.exp( -29.566 + 33.372*dS/(dS+2)  + 0.077/2 )# kg
        else:
            # Johansson 1999 -- note: this produces large leaf mass estimates for the upper range (12-13cm) 
            dlw_i = 0.00371*( dS*10 ) ** 1.11993 # kg
    return dlw_i

def slw( treespecies ):
    # specific leaf weight, g/m2
    # SLA as used by Majasalmi et al. 2013
    if treespecies == 1:
        # pine (Palmroth and Hari 2001)
        slw_i =  1000.0/6.2
    elif treespecies == 2:
        # spruce (Stenberg et al 1999)
        slw_i = 1000.0/4.95
    else:
        # birch (Lintunen et al. 2011) or other broadleaf
        slw_i = 1000.0/13.55
    return slw_i
    
def bailai( treespecies ):
    # branch area ratio to leaf area
    # as used by Majasalmi et al 2013
    if treespecies == 1:
        # pine (Stenberg et al 2003)
        bailai_i = 0.18
    elif treespecies == 2:
        # spruce (Jonckheere et al 2005)
        bailai_i = 0.18
    else:
        # birch or other broadleaf
        bailai_i = 0.15
    return bailai_i
    
def STAR( treespecies ):
    # spherically averaged shoot silhouette to total area ratio
    # STAR as used by Majasalmi et al. 2013
    if treespecies == 1:
        # pine (Smolander et al. 1994)
        STAR_i = 0.147
    elif treespecies == 2:
        # spruce (Stenberg et al. 1995)
        STAR_i = 0.161
    else:
        # birch or other broadleaf (no shoot)
        STAR_i = 0.25
    return STAR_i
    
def crownradius( treespecies, trunkdiameter, model="NS" ):
    # crown radius of a tree
    # crownradius as used by Majasalmi et al. 2013: Jakobsons 1970, Northern Sweden (Southern also available)
    dS = 2+1.25*trunkdiameter
    if treespecies == 1:
        # pine
        if model == "SS":
            crownradius_i = (1.14*dS+10.7)/20 # Jakobsons 1970, South Sweden
        else:
            crownradius_i = (1.19*dS+8.0)/20 # Jakobsons 1970, North Sweden
    elif treespecies == 2:
        # spruce
        if model == "SS":
            crownradius_i = (1.06*dS+15.5)/20 # Jakobsons 1970, South Sweden
        else:
            crownradius_i = (0.96*dS+13.3)/20 # Jakobsons 1970, North Sweden
    else:
        # birch or other broadleaf
        if model == "SS":
            crownradius_i = (1.51*dS + 12.8 )/20 # Jakobsons 1970, South Sweden
        else:
            crownradius_i = (1.40*dS + 12.2 )/20 # Jakobsons 1970, North Sweden
    return crownradius_i
