import os
import pickle
from typing import Tuple, List
import xml.etree.ElementTree as ETree

import numpy as np
from skimage import draw
from skimage.io import imread
from skimage.measure import label
from skimage.morphology import disk, dilation

import openslide


class ClassLabels:
    class_labels = {
        "epithelial": {"channel": 0, "classid": 1, "color": np.array([255, 0, 0])},
        "lymphocyte": {"channel": 1, "classid": 2, "color": np.array([255, 255, 0])},
        "neutrophil": {"channel": 2, "classid": 3, "color": np.array([0, 0, 255])},
        "macrophage": {"channel": 3, "classid": 4, "color": np.array([0, 255, 0])},
        "ambiguous": {"channel": 4, "classid": 5, "color": None},
        "border": {"channel": None, "classid": None, "color": np.array([139, 69, 19])}
    }
    n_channels = 5
    n_classes = 4

    @classmethod
    def get_color(cls, label_name: str) -> np.array:
        """Get color in the color-coded prediction masks"""
        if label_name not in cls.class_labels:
            raise ValueError(f"Unknown label: {label_name}")
        return cls.class_labels[label_name]["color"]

    @classmethod
    def get_channel(cls, label_name: str) -> np.array:
        """Get class channel in the nary mask "1-class-per-channel" encoding."""
        if label_name not in cls.class_labels:
            raise ValueError(f"Unknown label: {label_name}")
        return cls.class_labels[label_name]["channel"]

    @classmethod
    def get_classid(cls, label_name: str) -> np.array:
        """Get class id in the nary mask "uid/class" encoding."""
        if label_name not in cls.class_labels:
            raise ValueError(f"Unknown label: {label_name}")
        return cls.class_labels[label_name]["classid"]

    @classmethod
    def get_class_iterator(cls) -> list:
        return ["epithelial", "lymphocyte", "neutrophil", "macrophage"]

    @classmethod
    def get_class_colors_iterator(cls) -> list:
        return [cls.get_color(cl) for cl in cls.get_class_iterator()]

    @classmethod
    def get_label_from_channel(cls, channel):
        for key, val in cls.class_labels.items():
            if val["channel"] == channel:
                return key
        return None


class LabelCounter:
    """Class to count objects and surface, to make sure that we are working on the same data as the challenge."""
    count_objects = {}
    count_surface = {}

    @classmethod
    def getCountDict(cls) -> dict:
        """Get dictionnary with classes as keys and initialized to a value of 0"""
        count = {}
        for key in ClassLabels.class_labels:
            count[key] = 0
        return count

    @classmethod
    def reset_counts(cls):
        """Reset object & surface counts"""
        cls.count_objects = cls.getCountDict()
        cls.count_surface = cls.getCountDict()

    @classmethod
    def add_objects(cls, label_name: str, n):
        cls.count_objects[label_name] += n

    @classmethod
    def add_surface(cls, label_name: str, surface):
        cls.count_surface[label_name] += surface


class AnnotationPolygon:
    """Class for storing annotation polygons with a label"""

    def __init__(self, label_name, vertices):
        self.label = label_name
        self.vertices = vertices
        self.channel = ClassLabels.get_channel(label_name)


def get_xml_annotations(xml_file: str) -> List[AnnotationPolygon]:
    """Reads .xml annotations file & returns list of annotations"""
    annotations = []
    tree = ETree.parse(xml_file)
    root = tree.getroot()
    for attrs, regions, plots in root:
        label_name = attrs[0].attrib['Name']

        for region in regions:
            if region.tag == 'RegionAttributeHeaders':
                continue
            vertices = region[1]
            coords = np.array(
                [[int(float(vertex.attrib['X'])), int(float(vertex.attrib['Y']))] for vertex in vertices]).astype('int')

            annotations.append(AnnotationPolygon(label_name.lower(), coords))

    return annotations


def generate_mask(svs_file: str) -> np.array:
    """Generate n-ary mask from a slide's annotations."""
    wsi = openslide.OpenSlide(svs_file)
    size = wsi.level_dimensions[0]
    mask = np.zeros((size[1], size[0], ClassLabels.n_channels)).astype('int')
    annotations = get_xml_annotations(svs_file.replace('.svs', '.xml'))

    for idl, annotation in enumerate(annotations):
        fill = draw.polygon(annotation.vertices[:, 1], annotation.vertices[:, 0], mask.shape)
        mask[fill[0], fill[1], annotation.channel] = idl + 1

    return mask


def generate_masks(directory: str) -> Tuple[dict, dict]:
    """Generate n-ary masks (labeled objects w/ 1 channel per class) from a directory which follows the structure:
    - directory
        - patient folder
            - slide.svs
            - slide.svs
            - ...
        - ...
    
    Masks are saved into a pickle file as a dictionary {PATIENT : { SLIDE : np.array }}

    Returns the total number of objects per class and the surface in pixels per class found in the annotations.
    """
    patients = [f for f in os.listdir(directory) if
                f.startswith("TCGA")]  # remove .pkl from list if it's there already.

    print(f"{len(patients)} patients in directory: {directory}")

    LabelCounter.reset_counts()

    all_masks = {}

    for ip, patient in enumerate(patients):
        print(f"{ip + 1}/{len(patients)}", end="\r")

        patient_dir = os.path.join(directory, patient)
        slides = [f for f in os.listdir(patient_dir) if f.split('.')[1] == 'svs']
        all_masks[patient] = {}
        for slide in slides:
            mask = generate_mask(os.path.join(patient_dir, slide))
            for label_channel in range(ClassLabels.n_channels):
                label_name = ClassLabels.get_label_from_channel(label_channel)
                LabelCounter.add_objects(label_name, len(np.unique(mask[..., label_channel])) - 1)
                LabelCounter.add_surface(label_name, (mask[..., label_channel] > 0).sum())

            mask = relabel_nary_mask(mask)

            all_masks[patient][slide.replace('.svs', '')] = mask

    with open(os.path.join(directory, f"nary_masks.pkl"), "wb") as fp:
        pickle.dump(all_masks, fp)

    return LabelCounter.count_objects, LabelCounter.count_surface


def nary_from_colormap_no_border(cl_im: np.array) -> np.array:
    """Produce n-ary mask from the color-coded image by removing the borders and re-labeling the resulting objects."""
    mask_ids = np.zeros(cl_im.shape[:2] + (4,))

    masks = [(cl_im[..., 0] == cl[0]) * (cl_im[..., 1] == cl[1]) * (cl_im[..., 2] == cl[2]) for cl in
             ClassLabels.get_class_colors_iterator()]
    masks = [label(mask) for mask in masks]

    for i, m in enumerate(masks):
        mask_ids[..., i] = m

    return mask_ids


def nary_from_colormap_dilation(cl_im: np.array) -> np.array:
    """Produce n-ary mask from the color-coded image by removing the borders and re-labeling the resulting objects, 
    then trying to restore the borders by morphological dilation."""
    mask_ids = np.zeros(cl_im.shape[:2] + (4,))

    masks = [(cl_im[..., 0] == cl[0]) * (cl_im[..., 1] == cl[1]) * (cl_im[..., 2] == cl[2]) for cl in
             ClassLabels.get_class_colors_iterator()]
    masks = [label(mask) for mask in masks]

    # dilate each object
    for i in range(len(masks)):
        for j in range(1, masks[i].max() + 1):
            dilated_obj = dilation(masks[i] == j, disk(1))
            masks[i][dilated_obj] = j

    for i, m in enumerate(masks):
        mask_ids[..., i] = m

    return mask_ids


def generate_nary_from_colormap(directory: str, nary_fn, ext: str = '', *args) -> None:
    """Generates the nary masks from the colormap in the given directory, using a nary generating function.
    Extra arguments are meant to be passed to the nary_fn.
    
    Saves the results as a pickled dictionary {ext}_nary.pkl in directory"""
    all_masks = {}
    patients = [f for f in os.listdir(directory) if f.startswith("TCGA")]
    for ip, patient in enumerate(patients):
        print(f"Processing patient {ip + 1:2d}/{len(patients)}", end='\r')
        patient_dir = os.path.join(directory, patient)
        all_masks[patient] = {}
        images = [f for f in os.listdir(patient_dir) if 'mask' in f]
        for im in images:
            cc_mask = imread(os.path.join(patient_dir, im))
            nary_mask = relabel_nary_mask(nary_fn(cc_mask, *args))
            all_masks[patient][im.replace('_mask.png.tif', '').replace('_RGB_mask.tif.tif', '')] = nary_mask

    with open(os.path.join(directory, f"{ext}_nary.pkl"), "wb") as fp:
        pickle.dump(all_masks, fp)


def relabel_nary_mask(nary_mask: np.array) -> np.array:
    """Re-label the n-ary mask so that instead of the channels we have a 2-channels masks -> uid & class."""
    offset = 1
    nary_out = np.zeros(nary_mask.shape[:2] + (2,)).astype('int')
    for i in range(nary_mask.shape[2]):
        for j in np.unique(nary_mask[..., i]):
            if j == 0:
                continue
            nary_out[nary_mask[..., i] == j, 0] = offset
            offset += 1

        nary_out[nary_mask[..., i] > 0, 1] = i + 1

    return nary_out


def generate_nary_from_colorcoded(teams_dir: str, ccgt_dir: str):
    """Generate the n-ary masks from the color-coded predictions of all the teams using the border-removed
    and the border-dilated generation methods."""
    teams = os.listdir(teams_dir)
    for idt, team in enumerate(teams):
        print(f"Generating n-ary masks for team: {idt+1}")
        generate_nary_from_colormap(os.path.join(teams_dir, team), nary_from_colormap_no_border, 'border-removed')
        generate_nary_from_colormap(os.path.join(teams_dir, team), nary_from_colormap_dilation, 'border-dilated')

    print(f"Generating n-ary masks for the color-coded ground truth")
    generate_nary_from_colormap(ccgt_dir, nary_from_colormap_no_border, 'border-removed')
    generate_nary_from_colormap(ccgt_dir, nary_from_colormap_dilation, 'border-dilated')

