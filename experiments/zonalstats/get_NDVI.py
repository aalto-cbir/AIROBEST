"""
get_NDVI.py function computes NDVI values when given lists of reflectances from red and near infrared (NIR) bands.
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


OUTPUT:
Function returns list of NDVI values.

"""

def getndvi( red_band_list , nir_band_list ):
    
    # Append new NDVI values into a list
    NDVI_list = []
    
    for j in range(len(red_band_list)):
        RED  = red_band_list[j]
        NIR  = nir_band_list[j]
        NDVI = (NIR-RED) / (NIR+RED)
        NDVI_list.append(NDVI)

    return NDVI_list