import numpy as np
from Cell import Cell
from LabelList import LabelList

class ImageGrid(object):
    def __init__(self, image, gridWidth, gridHeight):
        self.img = image.copy()
        self.width = gridWidth
        self.height = gridHeight
        tempGrid = [np.hsplit(row, gridWidth) for row in np.vsplit(self.img, gridHeight)]
        self.grid = [[Cell(rowIdx, colIdx, img) for colIdx, img in enumerate(row)] for rowIdx, row in enumerate(tempGrid)]
        self.labelList = self.labelCells()
    
    def labelCells(self):
        labelList = LabelList()
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                if self.grid[row][col].label == None:
                    curLabel = labelList.getNewLabel()
                    self.recursiveLabel(row, col, curLabel)
        return labelList
    
    def calcAngleDiff(self, angle1, angle2):
        diff = abs(angle1 - angle2)
        diff = diff if diff <= 90 else (180.0 - diff)
        return diff
    
    def recursiveLabel(self, row, col, curLabel):
        gridHeight = len(self.grid)
        gridWidth = len(self.grid[0])
        angleThreshold = 3
        if row in range(gridHeight) and col in range(gridWidth):
            curCell = self.grid[row][col]
            if curCell.hasBarcodeFeatures and curCell.label == None:
                curLabel.assignTo(curCell)
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        newRow = row+i
                        newCol = col+j
                        if newRow in range(gridHeight) and newCol in range(gridWidth):
                            neighbor = self.grid[newRow][newCol]
                            angleDiff = self.calcAngleDiff(curCell.orientation, neighbor.orientation)
                            if angleDiff < angleThreshold:
                                self.recursiveLabel(newRow, newCol, curLabel)
    
    def getConcatGrid(self, imgGrid):
        return np.concatenate([np.concatenate(row, axis=1) for row in imgGrid], axis=0)
    
    def getGridImage(self):
        cellImgGrid = [[cell.img for cell in row]for row in self.grid]
        return self.getConcatGrid(cellImgGrid)
    
    def getLabelBorderImage(self):
        labelImgGrid = [[cell.getLabelBorderImage() for cell in row]for row in self.grid]
        return self.getConcatGrid(labelImgGrid)
    
    def getLabelOrientationImage(self):
        labelImgGrid = [[cell.getLabelOrientationImage() for cell in row]for row in self.grid]
        return self.getConcatGrid(labelImgGrid)
    
    def getLabelMaskImage(self, label):
        labelMaskGrid = [[cell.getLabelMaskImage(label) for cell in row]for row in self.grid]
        return self.getConcatGrid(labelMaskGrid)