import cv2, math
import numpy as np


class Utils(object):
    IMG_DIR = '/Users/leonardofilipe/PGC-img/'

    @staticmethod
    def get_contour_orientation(contour):
        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        if angle < -45:
            angle += 90
        return angle

        moments = cv2.moments(contour)
        mu00 = moments['m00']
        if mu00 == 0:
            return None
        mul11 = moments['mu11'] / mu00
        mul02 = moments['mu02'] / mu00
        mul20 = moments['mu20'] / mu00
        if (mul20 - mul02) == 0:
            return None
        try:
            angle_rad = 0.5 * math.atan((2 * mul11) / (mul20 - mul02))
            return math.degrees(angle_rad)
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
