import cv2, math
import numpy as np
from PIL import Image
import zbar
import io


class Utils(object):
    IMG_DIR = '/Users/leonardofilipe/PGC-img/'

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
    def get_cluster_mean(elements, labels):
        return np.mean(elements)
        #####################################
        if len(elements > 2):
            biggest_cluster = 0
            for i in range(len(labels)):
                if len(elements[labels==i]) > len(elements[labels==biggest_cluster]):
                    biggest_cluster = i
            return np.mean(elements[labels==biggest_cluster])
        else:
            return np.mean(elements)

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
