from ImageGrid import ImageGrid
from CommonUtils import Utils
import numpy as np
import cv2
import platform
import os
import glob
import time

CELL_SIZE = 40
PAGE_ROTATION = 0
PDF_PATH = ''

# Customize these paths according to your system before running
Utils.IMG_DIR = '/Users/leonardofilipe/PGC-img/'
PDF_PATH = '/Users/leonardofilipe/Dropbox/UFABC/PGC-Leonardo/codigo/corrections/'


def get_morph_barcode_rect(image):
    gradient = Utils.scharr_gradient(image)
    # blur and threshold the image
    thresh = Utils.otsu_binary(gradient)

    closed = Utils.morph_close(thresh, (21, 7))

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=6)
    closed = cv2.dilate(closed, None, iterations=6)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    (cntIm, contours, _) = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # compute the rotated bounding box of the largest contour
    return [cv2.minAreaRect(c) for c in contours]


def get_mctest_barcode_rect(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)

    img = cv2.erode(img, None, iterations=10)
    img = cv2.dilate(img, None, iterations=10)

    # find the contours in the thresholded image
    (cntIm, cnts, _) = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if no contours were found, return None
    if len(cnts) == 0:
        return []

    # otherwise, sort the contours by area and compute the rotated
    # bounding box of the largest contour
    contours = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    return [cv2.minAreaRect(c) for c in contours]


def mctest_extract_rect_img(image, rect):
    box = np.int0(cv2.boxPoints(rect))
    square = box
    y1, x1 = square[1]
    y2, x2 = square[3]
    [p1, p2] = [[min(x1, x2), min(y1, y2)], [max(x1, x2), max(y1, y2)]]
    bord = 5
    return image[p1[0] - bord:p2[0] + bord, p1[1] - bord:p2[1] + bord]


def get_morph_barcode_sub_imgs(image):
    rects = get_morph_barcode_rect(255-image)
    return [Utils.extract_rect_img(255-image, rect) for rect in rects]


def get_mctest_barcode_sub_imgs(image):
    rects = get_mctest_barcode_rect(255-image)
    return [mctest_extract_rect_img(255-image, rect) for rect in rects]


def get_subimg_barcode_sub_imgs(image):
    height, width = image.shape
    grid_height = int(height / CELL_SIZE)
    grid_width = int(width / CELL_SIZE)
    resize_shape = (grid_width*CELL_SIZE, grid_height*CELL_SIZE)
    grid_shape = (grid_width, grid_height)
    resized = cv2.resize(image, resize_shape)
    binary = Utils.otsu_binary(resized)
    rects = get_subimg_barcode_rects(binary, grid_shape)
    return [Utils.extract_rect_img(255-resized, rect) for rect in rects]


def get_subimg_barcode_rects(image, grid_dim):
    grid_height, grid_width = grid_dim
    grid = ImageGrid(image, grid_height, grid_width)
    labels = grid.labelList.get_list_sorted_by_size_desc()[:5]
    rects = [grid.get_label_rect(label) for label in labels]
    return filter(lambda x: x is not None, rects)


def test_segmentation_method(barcode_seg_func):
    total_tests = 0
    total_success = 0
    listext = ['*.pdf']
    listdir = glob.os.listdir(PDF_PATH)
    listdir.append('')
    for directory in listdir:
        for ext in listext:
            for file in np.sort(glob.glob(PDF_PATH + directory + '/' + ext)):
                for image in Utils.get_images_from_pdf(file):
                    # Rotate page if rotation is set
                    if PAGE_ROTATION != 0:
                        rows, cols = image.shape
                        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), PAGE_ROTATION, 1)
                        image = cv2.warpAffine(image, m, image.shape)
                    total_tests += 1
                    found_code = False
                    segments = barcode_seg_func(image)
                    for barcode_img in segments:
                        code = Utils.decode_barcode_img(barcode_img)
                        if code is not None:
                            found_code = True
                            break
                    if found_code:
                        total_success += 1
                    else:
                        filename = '{}{}_failed{:d}[C{:2d}][A{:2d}].png'.format(Utils.IMG_DIR,
                                                                                barcode_seg_func.__name__,
                                                                                total_tests, CELL_SIZE,
                                                                                PAGE_ROTATION)
                        cv2.imwrite(filename, image)
                        seg_num = 1
                        for segment in segments:
                            rows, cols = segment.shape
                            if rows < 1 or cols < 1:
                                continue
                            filename = '{}{}_failed{:d}[C{:2d}][A{:2d}][seg{:d}].png'.format(Utils.IMG_DIR,
                                                                                             barcode_seg_func.__name__,
                                                                                             total_tests, CELL_SIZE,
                                                                                             PAGE_ROTATION, seg_num)
                            cv2.imwrite(filename, segment)
                            seg_num += 1
    return total_tests, total_success


# Begin testing
print "      Python version: {}".format(platform.python_version())
print "      OpenCV version: {}".format(cv2.__version__)


for file in os.listdir(Utils.IMG_DIR):
    path = os.path.join(Utils.IMG_DIR, file)
    try:
        if os.path.isfile(path):
            os.unlink(path)
    except Exception as e:
        print e

start_time = time.time()
tests, success = test_segmentation_method(get_mctest_barcode_sub_imgs)
time_elapsed = time.time() - start_time
print '         MCTest test results: {:4d} out of {:4d} images in {:4.2f} s'.format(success, tests, time_elapsed)

start_time = time.time()
tests, success = test_segmentation_method(get_morph_barcode_sub_imgs)
time_elapsed = time.time() - start_time
print '  Morphological test results: {:4d} out of {:4d} images in {:4.2f} s'.format(success, tests, time_elapsed)

best_count = 0
for test_cell_size in [30, 40, 50]:
    CELL_SIZE = test_cell_size
    start_time = time.time()
    tests, success = test_segmentation_method(get_subimg_barcode_sub_imgs)
    time_elapsed = time.time() - start_time
    print ' Sub image[C{:2d}] test results: {:4d} out of {:4d} images in {:4.2f} s'.format(test_cell_size, success,
                                                                                           tests, time_elapsed)
    if success >= best_count:
        best_size = test_cell_size

CELL_SIZE = test_cell_size
for test_angle in [25, 45, 85, 90]:
    PAGE_ROTATION = test_angle
    start_time = time.time()
    tests, success = test_segmentation_method(get_mctest_barcode_sub_imgs)
    time_elapsed = time.time() - start_time
    print '    MCTest[A{:2d}] test results: {:4d} out of {:4d} images in {:4.2f} s'.format(test_angle, success, tests,
                                                                                           time_elapsed)

    start_time = time.time()
    tests, success = test_segmentation_method(get_morph_barcode_sub_imgs)
    time_elapsed = time.time() - start_time
    print ' Morpholog[A{:2d}] test results: {:4d} out of {:4d} images in {:4.2f} s'.format(test_angle, success,
                                                                                           tests, time_elapsed)

    start_time = time.time()
    tests, success = test_segmentation_method(get_subimg_barcode_sub_imgs)
    time_elapsed = time.time() - start_time
    print ' Sub image[A{:2d}] test results: {:4d} out of {:4d} images in {:4.2f} s'.format(test_angle, success,
                                                                                           tests, time_elapsed)
