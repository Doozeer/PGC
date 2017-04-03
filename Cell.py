import cv2
import math
import numpy as np

class Cell(object):
    def __init__(self, row, col, cellImg):
        self.img = cellImg
        self.row = row
        self.col = col
        self.hasBarcodeFeatures = False
        self.orientation = 0
        self.label = None
        self.contours = None
        self.evaluateBarcodeFeatures()
    
    def getImgContours(self):
        cntIm, cnts, hierarchy = cv2.findContours(self.img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = cnts
        return cnts
    
    def getContourImg(self):
        if self.contours == None:
            self.getImgContours()
        cellImg = cv2.cvtColor(255-self.img, cv2.COLOR_GRAY2RGB)
        return cv2.drawContours(cellImg, self.contours, -1, (0,255,0), 3)
    
    def getSkeletonizedImage(self):
        """ OpenCV function to return a skeletonized version of img, a Mat object"""
        #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
        img = self.img.copy() # don't clobber original
        skel = self.img.copy()
        skel[:,:] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        while True:
            eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
            temp  = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img[:,:] = eroded[:,:]
            if cv2.countNonZero(img) == 0 or cv2.countNonZero(img) == img.size:
                break
        return skel
    
    def getSkeletonizedImgContours(self):
        skeletonized = self.getSkeletonizedImage()
        cntIm, cnts, hierarchy = cv2.findContours(skeletonized, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = cnts
        return cnts
    
    def getContourOrientation(self, contour):
        moments = cv2.moments(contour)
        mu00 = moments['m00']
        mul11 = moments['mu11'] / mu00
        mul02 = moments['mu02'] / mu00
        mul20 = moments['mu20'] / mu00
        if (mul20 - mul02) == 0:
            return None
        try:
            angleRad = 0.5 * math.atan((2 * mul11) / (mul20 - mul02))
            return math.degrees(angleRad)
        except:
            print contour
            raise
    
    def evaluateBarcodeFeatures(self):
        if self.contours == None:
            self.getImgContours()
        
        # Filter out smaller contours which should be random noise
        sizeThresh = 30
        filteredContours = [contour for contour in self.contours if cv2.moments(contour)['m00'] > sizeThresh]
        
        # Get contour orientations
        angles = [x for x in [self.getContourOrientation(contour) for contour in filteredContours] if x != None]
        
        # Eliminate cells with a small number of valid contours
        nElements = len(angles)
        if nElements < 2:
             # cell is not part of barcode
            self.hasBarcodeFeatures = False
            return self
        
        # Use K-means clustering to determine if there is a prevalent orientation among contours
        # Define criteria = ( type, max_iter = 10 , epsilon = 0.5 )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5)
        flags = cv2.KMEANS_RANDOM_CENTERS
        clusterElements = np.float32(angles)
        compactness, labels, centers = cv2.kmeans(clusterElements,2,None,criteria,10,flags)
        
        # Smaller compactness should occur for a group of similarly oriented contours
        meanCompactness = compactness / float(len(clusterElements))
        compactnessThreshold = 1
        if meanCompactness < compactnessThreshold:
            self.hasBarcodeFeatures = True
            self.orientation = np.mean(clusterElements)
            return self
        else:
            self.hasBarcodeFeatures = False
            return self
    
    def getImageWithBorder(self, borderColor = (127, 127, 127)):
        cellImg = cv2.cvtColor(255-self.img, cv2.COLOR_GRAY2RGB)
        if borderColor == None:
            return cellImg
        cellImg[0:2,:]  = borderColor
        cellImg[:,0:2]  = borderColor
        cellImg[-2:,:] = borderColor
        cellImg[:,-2:] = borderColor
        return cellImg
    
    def getFillImage(self, fillColor = (127, 127, 127)):
        cellImg = cv2.cvtColor(255-self.img, cv2.COLOR_GRAY2RGB)
        if fillColor == None:
            return cellImg
        cellImg[:,:] = fillColor
        return cellImg
    
    def getLabelMaskImage(self, label):
        img = self.img.copy()
        if self.label == label:
            img[:,:] = 255
            return img
        else:
            img[:,:] = 0
            return img
    
    def getLabelBorderImage(self):
        if self.label != None:
            labelColor = self.label.color
            cellImg = self.getImageWithBorder(labelColor)
            return cellImg
        else:
            return self.getImageWithBorder(None)
    
    def getLabelOrientationImage(self):
        if self.label != None:
            labelColor = self.label.color
            cellImg = self.getFillImage(labelColor)
            font = cv2.FONT_HERSHEY_SIMPLEX
            textColor = (255-labelColor[0], 255-labelColor[1], 255-labelColor[2])
            cv2.putText(cellImg,str(self.orientation),(2,30), font, 0.6,textColor,2,cv2.LINE_AA)
            return cellImg
        else:
            return self.getFillImage(None)