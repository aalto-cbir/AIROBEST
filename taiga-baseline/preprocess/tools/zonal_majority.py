"""
zonal_majority.py
Copyright (C) 2019 Eelis Halme

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see http://www.gnu.org/licenses/

******************************************************************************************

    INPUT:   - Geometry (ogr geometry or WKT)
             - Raster image (as array with affine transformation)
             - Number of bands in the raster image
    OUTPUT:  - List of mean reflectance values within the given geometry.
             - If no data or too much no data inside the geometry, return -9999 (just an arbitrary number)
    
"""
from rasterstats import zonal_stats

def zonalmajority(my_geometry, my_raster, affine, number_of_bands):
    
    # Geometry to be used
    POLYGON = my_geometry
    
    # Raster image to be used
    RASTER_Array = my_raster
    
    # Create an empty list for the zonal statistics
    majority_list = []
    
    for i in range(number_of_bands):
        # Apply the zonal_stats function from rasterstats. Call for zonal mean and majority value.
        # Zonal_stats returns a list of dictionaries containing the stats. Set geojson_out to True in order to get only the value of interest.
        zonalstatistics = zonal_stats(POLYGON, RASTER_Array[i], affine=affine, band=1, geojson_out=True, stats=['mean' , 'majority'])
        
        # Zonal majority value: if this is zero then we don't have reflectance data within the geometry.
        majority_value = zonalstatistics[0]['properties']['majority']
        majority_list.append(majority_value)
        
    # If the majority value is zero for the band, there is no data.
    if majority_list.count(0) > 0:
        return -9999
    else:
        return majority_list