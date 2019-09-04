"""
get_buffers.py is a function to create buffers for geometries
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


Inputs:
    - list that includes one or more geometry
    - buffer size in meters
    
Output:
    - list that includes all new geometries

"""

def getbuffers( geometry_list, buffer_size ):
    
    # Append the new buffers to this empty list
    buffer_geometries = []
    
    for geom in geometry_list:
        buffer_geometry = geom.Buffer(buffer_size, 1)
        buffer_geometries.append(buffer_geometry)

    return buffer_geometries