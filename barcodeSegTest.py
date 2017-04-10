from ImageGrid import ImageGrid
from CommonUtils import Utils
import numpy as np
import cv2
import platform
import os
import glob
import time
import shutil

print "      Python version: {}".format(platform.python_version())
print "      OpenCV version: {}".format(cv2.__version__)

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


def get_morph_barcode_sub_imgs(image):
    rects = get_morph_barcode_rect(image)
    return [Utils.extract_rect_img(255-image, rect) for rect in rects]


def get_subimg_barcode_sub_imgs(image):
    cell_size = 40
    height, width = image.shape
    grid_height = int(height / cell_size)
    grid_width = int(width / cell_size)
    resize_shape = (grid_width*cell_size, grid_height*cell_size)
    grid_shape = (grid_width, grid_height)
    #resize_shape = (1632, 2368)
    #grid_shape = (51, 74)
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


for file in os.listdir(Utils.IMG_DIR):
    path = os.path.join(Utils.IMG_DIR, file)
    try:
        if os.path.isfile(path):
            os.unlink(path)
    except Exception as e:
        print e


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
                    total_tests += 1
                    found_code = False
                    for barcode_img in barcode_seg_func(image):
                        code = Utils.decode_barcode_img(barcode_img)
                        if code is not None:
                            found_code = True
                            break
                    if found_code:
                        total_success += 1
                    else:
                        filename = '{}{}_failed{:d}.png'.format(Utils.IMG_DIR, barcode_seg_func.__name__, total_tests)
                        cv2.imwrite(filename, image)
    return total_tests, total_success

start_time = time.time()
tests, success = test_segmentation_method(get_morph_barcode_sub_imgs)
time_elapsed = time.time() - start_time
print 'Morphological test results: {:4d} out of {:4d} images in {:4.2f} s'.format(success, tests, time_elapsed)

start_time = time.time()
tests, success = test_segmentation_method(get_subimg_barcode_sub_imgs)
time_elapsed = time.time() - start_time
print '    Sub image test results: {:4d} out of {:4d} images in {:4.2f} s'.format(success, tests, time_elapsed)

# pdf_images = Utils.get_images_from_pdf('test0.pdf')
#
# barcodeImg = get_morph_barcode_sub_imgs(pdf_images[0])[0]
# cv2.imwrite(Utils.IMG_DIR + 'barcodeMorph.jpg', barcodeImg)
# print Utils.decode_barcode_img(barcodeImg)
#
# i = 1
# barcode_imgs = get_subimg_barcode_sub_imgs(pdf_images[0])
# for img in barcode_imgs:
#    print Utils.decode_barcode_img(img)
#    cv2.imwrite(Utils.IMG_DIR + 'barcodeSub' + str(i) + '.jpg', img)
#    i += 1
