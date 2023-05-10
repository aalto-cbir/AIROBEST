#! /usr/bin/env python3

import spectral
import numpy as np

data_path          = '../../data/TAIGA'
taiga_source       = data_path+'/source'
forest_data_file   = taiga_source+'/forestdata_stands.hdr'
forest_data        = spectral.open_image(forest_data_file)
forest_data_full   = forest_data.open_memmap()
forest_data_mapped = np.array(forest_data_full, dtype='int')
band_names_raw     = np.array(forest_data.metadata['band names'])

b = None
for i, n in enumerate(band_names_raw):
    if n=='standID':
        b = i
        break
    
if b is None:
    print('standID band not found in', forest_data_file)

else:
    stand_ids           = forest_data_mapped[:, :, b]
    np.save(data_path + '/stand_ids.npy', stand_ids)

