# Path to Hila GeoPackage
Hila_data = ".gpkg"

# Path to Stand GeoPackage
Stand_data = ".gpkg"

# Path to a list of required stands
Stand_list = "required_stands_list.csv"

# Path to the raster image to be used (ENVI .hdr)
Raster_image = ".hdr"

# Name for the output csv-file (incl. Hila ID and Hila WKT geometry)
output_name = ".csv"

import main_function
main_function.main( Hila_data, Stand_data, Stand_list, Raster_image, output_name )