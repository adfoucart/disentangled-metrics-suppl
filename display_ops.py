import random
import os
from typing import Tuple
from matplotlib import pyplot as plt


def select_random_image(gt_dir: str) -> Tuple[str, str]:
    """Get a random sub-image from the directory. Returns patient & image name (without extension)."""
    patients = [f for f in os.listdir(gt_dir) if f.startswith("TCGA")]
    random.shuffle(patients)
    patient = patients.pop()
    patient_dir = os.path.join(gt_dir, patient)
    images = [f.replace('.svs', '') for f in os.listdir(patient_dir) if '.svs' in f]
    random.shuffle(images)
    im = images.pop()
    print(f"Selected image: {im}")
    return patient, im


def show_image_and_classes(rgb_image, nary, title='RGB'):
    """Display RGB, instances and classes for the given image."""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.title(title)
    plt.subplot(1, 3, 2)
    plt.imshow(nary[..., 0])
    plt.title('Instances')
    plt.subplot(1, 3, 3)
    plt.imshow(nary[..., 1], vmin=0, vmax=5)
    plt.title('Classes')
    plt.show()
