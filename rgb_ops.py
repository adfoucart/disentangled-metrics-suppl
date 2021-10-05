import os
import openslide
import numpy as np
from skimage.io import imread, imsave


def save_as_tif(svs_file: str) -> None:
    """Extract tif from svs file & save in same directory"""
    wsi = openslide.OpenSlide(svs_file)
    size = wsi.level_dimensions[0]
    rgb = np.array(wsi.read_region((0, 0), 0, wsi.level_dimensions[0]))
    imsave(svs_file.replace('.svs', '.tif'), rgb)


def extract_all_tifs(directory: str) -> None:
    """Browse all .svs files and convert to .tif"""
    patients = [f for f in os.listdir(directory) if f.startswith("TCGA")]
    for ip, patient in enumerate(patients):
        print(f"Processing patient {ip + 1}/{len(patients)}", end="\r")
        patient_dir = os.path.join(directory, patient)
        svs_files = [os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if f.endswith('.svs')]
        for svs in svs_files:
            save_as_tif(svs)
