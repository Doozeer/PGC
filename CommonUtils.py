import cv2
import math


class Utils(object):

    @staticmethod
    def get_contour_orientation(contour):
        moments = cv2.moments(contour)
        mu00 = moments['m00']
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
