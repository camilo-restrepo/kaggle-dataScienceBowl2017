import numpy as np
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time


INPUT_FOLDER = '/Volumes/Files/git/kaggle-dataScienceBowl2017/data/sample_images/'
OUTPUT_FOLDER = '/Volumes/Files/git/kaggle-dataScienceBowl2017/data/out-sample_images/'

hu_in_mean = 0
hu_resample_image_mean = 0
hu_out_image_mean = 0


def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path) if not s.startswith('.')]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    # current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # Thresholding
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Remove background
    background_label = labels[0, 0, 0]
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1
    binary_image = 1 - binary_image

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None:
        binary_image[labels != l_max] = 0

    return binary_image


def plot_3d(image, save_file, threshold=-300):
    # Position the scan upright
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.savefig(save_file + '.png', bbox_inches='tight')
    plt.close()


def process_file(patient):
    first_patient = load_scan(INPUT_FOLDER + patient)
    first_patient_pixels = get_pixels_hu(first_patient)
    hu_in_mean_local = np.mean(first_patient_pixels)

    pix_resampled, _ = resample(first_patient_pixels, first_patient, [1, 1, 1])
    hu_resample_image_mean_local = np.mean(pix_resampled)

    lungs_mask_fill = segment_lung_mask(pix_resampled, True)

    out_image = lungs_mask_fill * pix_resampled
    hu_out_image_mean_local = np.mean(out_image)

    # plot_3d(out_image, OUTPUT_FOLDER + patient)
    np.save(OUTPUT_FOLDER + patient, out_image)

    return hu_in_mean_local, hu_resample_image_mean_local, hu_out_image_mean_local


patients = os.listdir(INPUT_FOLDER)
patients.sort()
patients = [p for p in patients if not p.startswith(".")]
patients = patients[:4]

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

i = 1

start = time.time()
print('PROCESSING', len(patients), 'IMAGES...')
for p in patients:
    hu_in_tmp, hu_resample_tmp, hu_out_tmp = process_file(p)

    hu_in_mean += hu_in_tmp
    hu_resample_image_mean += hu_resample_tmp
    hu_out_image_mean += hu_out_tmp

    print(i, '-', round((i / len(patients)) * 100, 2), '% -', p)
    i += 1

stop = time.time()

hu_in_mean /= len(patients)
hu_resample_image_mean /= len(patients)
hu_out_image_mean /= len(patients)

print('HU in mean:', hu_in_mean, '\nHU resampled mean:', hu_resample_image_mean,
      '\nHU out mean:', hu_out_image_mean)
print('Total time:', stop - start, 'Mean time:', (stop - start) / len(patients))
