"""
Copyright (C) 2017,2018  Matti Mõttus 
This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.

Allometric functions for pine, spruce and birch for Finland
 treespecies numbering as common in forestry practice: 1=pine, 2=spruce 3=birch
   (can include other broadleaves), etc.
   
originally based on the models included in Majasalmi 2013 (with exception of crown radius and length)

------------ References ------------ 
A. Jakobsons 1970. The correlation between the diameter of tree crown and other tree factors — mainly the breastheight diameter , vol. 14. Sweden Department of Forest, Survey, Royal College of Forestry.
T. Johansson 1999. “Biomass equations for determining fractions of pendula and pubescent birches growing on abandoned farmland and some practical implications,” Biomass and Bioenergy , vol. 16, pp. 223–238
I. Jonckheere, B. Muys, and P. Coppin 2005. “Allometry and evaluation of in situ optical LAI determination in Scots pine: a case study in Belgium.,” Tree Physiology , vol. 25, pp. 723–732
A. Lintunen, R. Sievänen, P. Kaitaniemi, and J. Perttunen 2011. “Models of 3D crown structure for Scots pine (Pinus sylvestris) and silver birch (Betula pendula) grown in mixed forest,” Canadian Journal of Forest Research , vol. 41, pp. 1779–1794
T. Majasalmi, M. Rautiainen, P. Stenberg and P. Lukes 2013. “An assessment of ground reference methods for estimating LAI of boreal forests,” Forest Ecology and Management , vol. 292, pp. 10–18
T. Nilson, M. Lang, A. Kuusk, J. Anniste, and T. Lükk 1999. “Forest reflectance model as an interface between satellite images and forestry databases,” in Proceedings of IUFRO Conference on Remote Sensing and Monitoring, pp. 462–476.
S. Palmroth and P. Hari 2001. “Evaluation of the importance of acclimation of needle structure, photosynthesis, and respiration to available photosynthetically active radiation in a Scots pine canopy,” Canadian Journal of Forest Research , vol. 31, pp. 1235–1243
M. Rautiainen, M. Mõttus, P. Stenberg, and S. Ervasti, 2008 “Crown envelope shape measurements and models,” Silva Fennica , vol. 42, no. October 2007, pp. 19–33
J. Repola 2008. “Biomass equations for birch in Finland,” Silva Fennica , vol. 42, pp. 605–624
J. Repola 2009. “Biomass equations for Scots pine and Norway spruce in Finland,” Silva Fennica , vol. 43, no. 4, pp. 625–647
H. Smolander, P. Stenberg, and S. Linder 1994. “Dependence of light interception efficiency of Scots pine shoots on structural parameters,” Tree Physiology , vol. 14, no. 7–9, pp. 971–980
P. Stenberg, S. Linder, and H. Smolander 1995. “Variation in the ratio of shoot silhouette area to needle area in fertilized and unfertilized Norway spruce trees,” Tree Physiology , vol. 15, no. 11, pp. 705–712
P. Stenberg, T. Kangas, H. Smolander, and S. Linder 1999. “Shoot structure, canopy openness, and light interception in Norway spruce,” Plant Cell and Environment , vol. 22, no. 9, pp. 1133–1142
P. Stenberg, T. Nilson, H. Smolander, and P. Voipio 2003. “Gap fraction based estimation of LAI in Scots pine stands subjected to experimental removal of branches and stems,” Canadian Journal Of Remote Sensing , vol. 29, no. 3, pp. 363–370
"""

import numpy as np


def dlw( treespecies, trunkdiameter, treeheight ):
    """dry leaf weight, kg per tree
    in: tree height in m, tree trunk diameter (dbh) in cm.
     Repola (2008,2009), models based on inventory data only: dbh and height
     pine: Repola 2009 model 1a, Eq. (7)
     spruce: Repola 2009 model 1c, Eq. (15)
     birch: Repola 2008 separate model I, Eq. (12)
       for dbh<13.7 cm, Johansson's (1999) model as given by Majasalmi et al. 2013, Eq. (8)"""
    h = treeheight
    dS = 2+1.25*trunkdiameter
    if treespecies == 1:
        # pine
        dlw_i = np.exp( -6.303 + 14.472*dS/(dS+6) -3.976*h/(h+1) + (0.109+0.118)/2 ) 
    elif treespecies == 2:
        # spruce
        dlw_i = np.exp( -2.994 + 12.251*dS/(dS+10) -3.415*h/(h+1) +  (0.107+0.089)/2 )
    else:
        # birch or other broadleaf
        if trunkdiameter > 11:
            # Repola (2008) had only large trees in the sample
            dlw_i = np.exp( -29.566 + 33.372*dS/(dS+2)  + 0.077/2 )# kg
        else:
            # Johansson 1999 -- note: this produces large leaf mass estimates 
            #   for the upper range (12-13cm) 
            dlw_i = 0.00371*( dS*10 ) ** 1.11993 # kg
    return dlw_i

def slw( treespecies ):
    """Specific leaf weight, g/m2
    SLA as used by Majasalmi et al. 2013
    Note: Nilson et al. 1999 proposed 133, 152, 70.4 g/m2 for pine, psuce, birch, respectively"""
    if treespecies == 1:
        # pine (Palmroth and Hari 2001)
        slw_i =  161.3 # = 1000.0/6.2
    elif treespecies == 2:
        # spruce (Stenberg et al 1999)
        slw_i = 202.0 # = 1000.0/4.95
    else:
        # birch (Lintunen et al. 2011) or other broadleaf
        slw_i = 74.07 # = 1000.0/13.55
    return slw_i
    
def bailai( treespecies ):
    """branch area ratio to leaf area as used by Majasalmi et al 2013
    Note: Nilson et al. 1999 proposed the values 0.18, 0.12, 0.10 for pine, spruce, birch, respectively"""
    if treespecies == 1:
        # pine (Stenberg et al 2003)
        bailai_i = 0.18
    elif treespecies == 2:
        # spruce (Jonckheere et al 2005)
        bailai_i = 0.18
    else:
        # birch or other broadleaf
        bailai_i = 0.15 # Majasalmi et al. 2013 gave no reference for birch
    return bailai_i
    
def STAR( treespecies ):
    """spherically averaged shoot silhouette to total area ratio
    STAR as used by Majasalmi et al. 2013
    Note: Nilson et al. 1999 used 0.14, 0.15, 0.20 for pine, spruce, birch, respectively
    Note: The "shoot shading coefficient" in FRT or clumping index equals 4*STAR
    """
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
    
def crownradius( treespecies, trunkdiameter, treeheight=None, model="NS" ):
    """crown radius of a tree in meters
    trunkdiameter: dbh in cm
    treeheight: tree height in m, only used for some models
    model: NS: Jakobsons 1970, Northern Sweden
      SS: Jakobsons 1970, Southern Sweden
      Rautianen2008 for Norway spruce and Scots pine only -- 
        Note: DBH in table 2 is given in mm (not cm as given it caption) as is 
        evident when comparing crown volume with Fig. 2."""
    dS = 2+1.25*trunkdiameter # convert to diameter at base, for Jakobsons 1970
    if treespecies == 1:
        # pine
        if model == "SS":
            crownradius_i = (1.14*dS+10.7)/20 # Jakobsons 1970, South Sweden
        elif model == "NS":
            crownradius_i = (1.19*dS+8.0)/20 # Jakobsons 1970, North Sweden
        elif model == "Rautiainen2008":
            if treeheight is None:
                crownradius_i = 0.476 + 0.06*trunkdiameter
            else:
                crownradius_i = 0.700 + 0.10*trunkdiameter - 0.06*treeheight
        elif model == "Nilson1999":
            # the original model was given for 2R, divide coefficients by 2
            crownradius_i = 0.053*trunkdiameter + 0.3075*trunkdiameter/treeheight

    elif treespecies == 2:
        # spruce
        if model == "SS":
            crownradius_i = (1.06*dS+15.5)/20 # Jakobsons 1970, South Sweden
        elif model=="NS":
            crownradius_i = (0.96*dS+13.3)/20 # Jakobsons 1970, North Sweden
        elif model=="Rautiainen2008":
            if treeheight is None:
                crownradius_i = 0.663 + 0.06*trunkdiameter
            else:
                crownradius_i = 0.796 + 0.09*trunkdiameter - 0.05*treeheight
        elif model == "Nilson1999":
            # the original model was given for 2R, divide coefficients by 2
            crownradius_i = 0.0415*trunkdiameter + 0.53075*trunkdiameter/treeheight

    else:
        # birch or other broadleaf
        if model == "SS":
            crownradius_i = (1.51*dS + 12.8 )/20 # Jakobsons 1970, South Sweden
        elif model=="NS":
            crownradius_i = (1.40*dS + 12.2 )/20 # Jakobsons 1970, North Sweden
        elif model == "Nilson1999":
            # the original model was given for 2R, divide coefficients by 2
            crownradius_i = 0.067*trunkdiameter + 0.473*trunkdiameter/treeheight

    return crownradius_i

def crownlength( treespecies, trunkdiameter, treeheight=None, model="default" ):
    """ Crown length in meters
    model: Rautiainen2008 for Norway spruce and Scots pine only
        Note: DBH in table 2 is given in mm (not cm as given it caption) as is 
        evident when comparing crown volume with Fig. 2.
    trunkdiameter: dbh in cm
    treeheight: tree height in m, only used for some models
    default model -- pine and spruce from Rautiainen et al. 2008, birch from Nilson et al. 2003"""
    if model == "default":
        if treespecies < 3:
            model = "Rautiainen2008"
        else:
            model = "Nilson1999"
    if treespecies == 1:
        # pine
        if model == "Rautiainen2008":
            crownlength_i = 2.653 + 0.26*trunkdiameter
        else:
            # model=="Nilson1999"
            crownlength_i = 0.1474*treeheight/trunkdiameter*np.exp(1.2039*np.log(trunkdiameter))
    elif treespecies == 2:
        #spruce
        if model == "Rautiainen2008":
            crownlength_i = 2.686 + 0.52*trunkdiameter
        else:
            # model=="Nilson1999"
            crownlength_i = 1.5690*treeheight/trunkdiameter + np.exp(0.8235*np.log(trunkdiameter))
    else:
        # birch or other
        # model=="Nilson1999"
        crownlength_i = 0.5525*treeheight/trunkdiameter + np.exp(0.7032*np.log(trunkdiameter))
    return crownlength_i
    
def crownvolume_ellipsoid( treespecies, trunkdiameter, treeheight=None, model=None ):
    """ A helper function to calculate ellipsoid volume for crown (m3)
    """
    if model is None:
        r = crownradius( treespecies, trunkdiameter, treeheight )
        h = crownlength( treespecies, trunkdiameter, treeheight )
    else:
        r = crownradius( treespecies, trunkdiameter, treeheight, model=model )
        h = crownlength( treespecies, trunkdiameter, treeheight, model=model )
    return 4.0/3.0 * np.pi * h/2 * r**2

