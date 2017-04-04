from ImageGrid import ImageGrid
import cv2
import PyPDF2 as pypdf
import numpy as np
import platform
import random as rnd
print "      Python version: {}".format(platform.python_version())
print "      OpenCV version: {}".format(cv2.__version__)
print "     PyPDF2 version : {}".format(pypdf.__version__)

try:
    CV_CUR_LOAD_IM_GRAY = cv2.CV_LOAD_IMAGE_GRAYSCALE
except AttributeError:
    CV_CUR_LOAD_IM_GRAY = cv2.IMREAD_GRAYSCALE

def tiff_header_for_CCITT(width, height, img_size, CCITT_group=4):
    tiff_header_struct = '<' + '2s' + 'h' + 'l' + 'h' + 'hhll' * 8 + 'h'
    return struct.pack(tiff_header_struct,
                       b'II',  # Byte order indication: Little indian
                       42,  # Version number (always 42)
                       8,  # Offset to first IFD
                       8,  # Number of tags in IFD
                       256, 4, 1, width,  # ImageWidth, LONG, 1, width
                       257, 4, 1, height,  # ImageLength, LONG, 1, lenght
                       258, 3, 1, 1,  # BitsPerSample, SHORT, 1, 1
                       259, 3, 1, CCITT_group,  # Compression, SHORT, 1, 4 = CCITT Group 4 fax encoding
                       262, 3, 1, 0,  # Threshholding, SHORT, 1, 0 = WhiteIsZero
                       273, 4, 1, struct.calcsize(tiff_header_struct),  # StripOffsets, LONG, 1, len of header
                       278, 4, 1, height,  # RowsPerStrip, LONG, 1, lenght
                       279, 4, 1, img_size,  # StripByteCounts, LONG, 1, size of image
                       0  # last IFD
                       )

def handleCCITTFaxDecodeImg(obj):
    if obj['/DecodeParms']['/K'] == -1:
        CCITT_group = 4
    else:
        CCITT_group = 3
    width = obj['/Width']
    height = obj['/Height']
    data = obj._data  # sorry, getData() does not work for CCITTFaxDecode
    img_size = len(data)
    tiff_header = tiff_header_for_CCITT(width, height, img_size, CCITT_group)
    data = tiff_header + data
    return cv2.imdecode(np.frombuffer(data, np.uint8), CV_CUR_LOAD_IM_GRAY)

def handleOtherImg(obj):
    data = obj._data
    return 255-cv2.imdecode(np.frombuffer(data, np.uint8), CV_CUR_LOAD_IM_GRAY)

def getImgFromPage(pdfObj, page):
    colorSpaceDict = {}
    pageObj = pdfObj.getPage(page)
    xObj = pageObj['/Resources']['/XObject'].getObject()
    for obj in xObj:
        if xObj[obj]['/Subtype'] == '/Image':
            if xObj[obj]['/Filter'] == '/CCITTFaxDecode':
                return handleCCITTFaxDecodeImg(xObj[obj])
            else:
                return handleOtherImg(xObj[obj])

def getImagesFromPDF(filePath):
    pdfObj = pypdf.PdfFileReader(open(filePath, "rb"))
    nPages = pdfObj.getNumPages()
    images = [getImgFromPage(pdfObj, page) for page in range(nPages)]
    return images

def scharrGradient(image):
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction
    gradX = cv2.Scharr(image, ddepth = cv2.CV_32F, dx = 1, dy = 0)
    gradY = cv2.Scharr(image, ddepth = cv2.CV_32F, dx = 0, dy = 1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    return gradient

def otsuBinary(image):
    # Otsu's thresholding after Gaussian filtering
    blurred = cv2.GaussianBlur(image,(5,5),0)
    threshVal, threshImg = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshImg

def morphClose(image, kernelSize):
    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed

def extractRectImg(image, rect):
    widthPadding = 50
    box = cv2.boxPoints(rect) 
    box = np.int0(box)

    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    angle = rect[2]
    if angle < -45:
        angle += 90

    # Center of rectangle in source image
    center = ((x1+x2)/2,(y1+y2)/2)
    # Size of the upright rectangle bounding the rotated rectangle
    size = (x2-x1+widthPadding, y2-y1)
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(image, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = H if H > W else W
    croppedH = H if H < W else W
    # Final cropped & rotated rectangle
    return cv2.getRectSubPix(cropped, (int(croppedW)+widthPadding,int(croppedH)), (size[0]/2, size[1]/2))

def getMorphBarcodeRect(image):
    gradient = scharrGradient(image)
    # blur and threshold the image
    thresh = otsuBinary(gradient)

    closed = morphClose(thresh, (21, 7))

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 6)
    closed = cv2.dilate(closed, None, iterations = 6)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    (cntIm, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

    # compute the rotated bounding box of the largest contour
    return cv2.minAreaRect(c)

def getMorphBarcodeSubImg(image):
    rect = getMorphBarcodeRect(image)
    return extractRectImg(image, rect)

images = getImagesFromPDF('test0.pdf')
barcodeImg = getMorphBarcodeSubImg(images[0])

cv2.imwrite('barcode.jpg', barcodeImg)