import cv2, math
import numpy as np
from PIL import Image
import PyPDF2 as Pypdf
import zbar
import io
import struct


class Utils(object):
    IMG_DIR = '/Users/leonardofilipe/PGC-img/'

    try:
        CV_CUR_LOAD_IM_GRAY = cv2.CV_LOAD_IMAGE_GRAYSCALE
    except AttributeError:
        CV_CUR_LOAD_IM_GRAY = cv2.IMREAD_GRAYSCALE

    @staticmethod
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

    @staticmethod
    def handle_ccitt_fax_decode_img(obj):
        if obj['/DecodeParms']['/K'] == -1:
            ccitt_group = 4
        else:
            ccitt_group = 3
        width = obj['/Width']
        height = obj['/Height']
        data = obj._data  # sorry, getData() does not work for CCITTFaxDecode
        img_size = len(data)
        tiff_header = Utils.tiff_header_for_ccitt(width, height, img_size, ccitt_group)
        data = tiff_header + data
        return cv2.imdecode(np.frombuffer(data, np.uint8), Utils.CV_CUR_LOAD_IM_GRAY)

    @staticmethod
    def handle_other_img(obj):
        data = obj._data
        return 255 - cv2.imdecode(np.frombuffer(data, np.uint8), Utils.CV_CUR_LOAD_IM_GRAY)

    @staticmethod
    def get_img_from_page(pdf_obj, page):
        page_obj = pdf_obj.getPage(page)
        x_obj = page_obj['/Resources']['/XObject'].getObject()
        for obj in x_obj:
            if x_obj[obj]['/Subtype'] == '/Image':
                if x_obj[obj]['/Filter'] == '/CCITTFaxDecode':
                    return Utils.handle_ccitt_fax_decode_img(x_obj[obj])
                else:
                    return Utils.handle_other_img(x_obj[obj])

    @staticmethod
    def get_images_from_pdf(file_path):
        pdf_obj = Pypdf.PdfFileReader(open(file_path, "rb"))
        n_pages = pdf_obj.getNumPages()
        images = [Utils.get_img_from_page(pdf_obj, page) for page in range(n_pages)]
        return images

    @staticmethod
    def get_contour_orientation(contour):
        moments = cv2.moments(contour)
        mu11 = moments['mu11']
        mu02 = moments['mu02']
        mu20 = moments['mu20']
        if (mu20 - mu02) == 0:
            return None
        try:
            angle_rad = 0.5 * math.atan((2 * mu11) / (mu20 - mu02))
            angle_deg = math.degrees(angle_rad)
            return angle_deg
        except:
            print contour
            raise

    @staticmethod
    def calc_angle_diff(angle1, angle2):
        diff = abs(angle1 - angle2)
        diff = diff if diff <= 90 else (180.0 - diff)
        return diff

    @staticmethod
    def get_contour_size(contour):
        return cv2.moments(contour)['m00']

    @staticmethod
    def otsu_binary(image):
        # Otsu's thresholding after Gaussian filtering
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        thresh_val, thresh_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh_img

    @staticmethod
    def scharr_gradient(image):
        # compute the Scharr gradient magnitude representation of the images
        # in both the x and y direction
        grad_x = cv2.Scharr(image, ddepth=cv2.CV_32F, dx=1, dy=0)
        grad_y = cv2.Scharr(image, ddepth=cv2.CV_32F, dx=0, dy=1)

        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(grad_x, grad_y)
        gradient = cv2.convertScaleAbs(gradient)
        return gradient

    @staticmethod
    def morph_close(image, kernel_size):
        # construct a closing kernel and apply it to the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return closed

    @staticmethod
    def get_cluster_mean(elements, labels):
        return np.mean(elements)
        # disabled code below used for tests
        if len(elements > 2):
            biggest_cluster = 0
            for i in range(len(labels)):
                if len(elements[labels==i]) > len(elements[labels==biggest_cluster]):
                    biggest_cluster = i
            return np.mean(elements[labels==biggest_cluster])
        else:
            return np.mean(elements)

    @staticmethod
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
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Size of the upright rectangle bounding the rotated rectangle
        size = (x2 - x1 + width_padding, y2 - y1)

        m = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

        # Cropped upright rectangle
        cropped = cv2.getRectSubPix(image, size, center)
        cropped = cv2.warpAffine(cropped, m, size)
        cropped_w = h if h > w else w
        cropped_h = h if h < w else w

        # Final cropped & rotated rectangle
        return cv2.getRectSubPix(cropped, (int(cropped_w) + width_padding, int(cropped_h)), (size[0] / 2, size[1] / 2))

    @staticmethod
    def decode_barcode_img(cv2_img):
        try:
            _, cv2_img = cv2.threshold(cv2_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            r, png_bytes = cv2.imencode('.png', cv2_img)
            pil_png = Image.open(io.BytesIO(png_bytes)).convert('L')
            width, height = pil_png.size
            png_raw = pil_png.tostring()
            scanner = zbar.ImageScanner()
            scanner.parse_config('enable')
            image = zbar.Image(width, height, 'Y800', png_raw)
            scanner.scan(image)
            symbols = image.symbols
            for symbol in symbols:
                return symbol.data
            else:
                return None
        except Exception as e:
            print e.message
            return None
