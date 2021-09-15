import spectral
import pandas as pd
import numpy as np

data_path = '../../data/'
stand_id_data    = spectral.open_image('/scratch/project_2001284/hyperspectral/standids_in_pixels.hdr')
stand_ids_full   = stand_id_data.open_memmap()
stand_ids_mapped = np.array(stand_ids_full, dtype='int')
stand_ids_mapped = np.squeeze(stand_ids_mapped)  # remove the single-dimensional entries, size RxC
np.save(data_path + 'stand_ids.npy', stand_ids_mapped)