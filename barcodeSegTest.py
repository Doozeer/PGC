from ImageGrid import ImageGrid
from CommonUtils import Utils
import cv2
import PyPDF2 as Pypdf
import numpy as np
import platform
import struct
import os, shutil

print "      Python version: {}".format(platform.python_version())
print "      OpenCV version: {}".format(cv2.__version__)
print "     PyPDF2 version : {}".format(Pypdf.__version__)

try:
    CV_CUR_LOAD_IM_GRAY = cv2.CV_LOAD_IMAGE_GRAYSCALE
except AttributeError:
    CV_CUR_LOAD_IM_GRAY = cv2.IMREAD_GRAYSCALE


def tiff_header_for_ccitt(width, height, img_size, ccitt_group=4):
    tiff_header_struct = '<' + '2s' + 'h' + 'l' + 'h' + 'hhll' * 8 + 'h'
    return struct.pack(tiff_header_struct,
                       b'II',  # Byte order indication: Little indian
                       42,  # Version number (always 42)
                       8,  # Offset to first IFD
                       8,  # Number of tags in IFD
                       256, 4, 1, width,  # ImageWidth, LONG, 1, width
                       257, 4, 1, height,  # ImageLength, LONG, 1, lenght
                       258, 3, 1, 1,  # BitsPerSample, SHORT, 1, 1
                       259, 3, 1, ccitt_group,  # Compression, SHORT, 1, 4 = CCITT Group 4 fax encoding
                       262, 3, 1, 0,  # Threshholding, SHORT, 1, 0 = WhiteIsZero
                       273, 4, 1, struct.calcsize(tiff_header_struct),  # StripOffsets, LONG, 1, len of header
                       278, 4, 1, height,  # RowsPerStrip, LONG, 1, lenght
                       279, 4, 1, img_size,  # StripByteCounts, LONG, 1, size of image
                       0  # last IFD
                       )


def handle_ccitt_fax_decode_img(obj):
    if obj['/DecodeParms']['/K'] == -1:
        ccitt_group = 4
    else:
        ccitt_group = 3
    width = obj['/Width']
    height = obj['/Height']
    data = obj._data  # sorry, getData() does not work for CCITTFaxDecode
    img_size = len(data)
    tiff_header = tiff_header_for_ccitt(width, height, img_size, ccitt_group)
    data = tiff_header + data
    return cv2.imdecode(np.frombuffer(data, np.uint8), CV_CUR_LOAD_IM_GRAY)


def handle_other_img(obj):
    data = obj._data
    return 255-cv2.imdecode(np.frombuffer(data, np.uint8), CV_CUR_LOAD_IM_GRAY)


def get_img_from_page(pdf_obj, page):
    page_obj = pdf_obj.getPage(page)
    x_obj = page_obj['/Resources']['/XObject'].getObject()
    for obj in x_obj:
        if x_obj[obj]['/Subtype'] == '/Image':
            if x_obj[obj]['/Filter'] == '/CCITTFaxDecode':
                return handle_ccitt_fax_decode_img(x_obj[obj])
            else:
                return handle_other_img(x_obj[obj])


def get_images_from_pdf(file_path):
    pdf_obj = Pypdf.PdfFileReader(open(file_path, "rb"))
    n_pages = pdf_obj.getNumPages()
    images = [get_img_from_page(pdf_obj, page) for page in range(n_pages)]
    return images


def scharr_gradient(image):
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    grad_x = cv2.Scharr(image, ddepth=cv2.CV_32F, dx=1, dy=0)
    grad_y = cv2.Scharr(image, ddepth=cv2.CV_32F, dx=0, dy=1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)
    return gradient


def morph_close(image, kernel_size):
    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed


def extract_rect_img(image, rect):
    width_padding = 50
    box = cv2.boxPoints(rect) 
    box = np.int0(box)

    w = rect[1][0]
    h = rect[1][1]

    xs = [i[0] for i in box]
    ys = [i[1] for i in box]
    x1 = min(xs)
    x2 = max(xs)
    y1 = min(ys)
    y2 = max(ys)

    angle = rect[2]
    if angle < -45:
        angle += 90

    # Center of rectangle in source image
    center = ((x1+x2)/2, (y1+y2)/2)

    # Size of the upright rectangle bounding the rotated rectangle
    size = (x2-x1+width_padding, y2-y1)

    m = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(image, size, center)
    cropped = cv2.warpAffine(cropped, m, size)
    cropped_w = h if h > w else w
    cropped_h = h if h < w else w

    # Final cropped & rotated rectangle
    return cv2.getRectSubPix(cropped, (int(cropped_w)+width_padding,int(cropped_h)), (size[0]/2, size[1]/2))


def get_morph_barcode_rect(image):
    gradient = scharr_gradient(image)
    # blur and threshold the image
    thresh = Utils.otsu_binary(gradient)

    closed = morph_close(thresh, (21, 7))

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
    return [extract_rect_img(image, rect) for rect in rects]


def get_subimg_barcode_sub_imgs(image):
    resize_shape = (1632, 2368)
    grid_shape = (51, 74)
    resized = cv2.resize(image, resize_shape)
    resized = Utils.otsu_binary(resized)
    rects = get_subimg_barcode_rects(resized, grid_shape)
    return [extract_rect_img(resized, rect) for rect in rects]


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

pdf_images = get_images_from_pdf('test0.pdf')

barcodeImg = get_morph_barcode_sub_imgs(pdf_images[0])[0]
cv2.imwrite(Utils.IMG_DIR + 'barcodeMorph.jpg', barcodeImg)

i = 1
barcode_imgs = get_subimg_barcode_sub_imgs(pdf_images[0])
for img in barcode_imgs:
    cv2.imwrite(Utils.IMG_DIR + 'barcodeSub' + str(i) + '.jpg', img)
    i += 1


