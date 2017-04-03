from Label import Label

class LabelList(object):
    def __init__(self):
        self.labelList = []
    
    def getLabelCount(self):
        return len(self.labelList)
    
    def getNewLabel(self):
        newLabel = Label(self.getLabelCount(), self)
        self.labelList.append(newLabel)
        return newLabel
    
    def getCurLabel(self):
        if len(self.labelList) < 1:
            self.getNewLabel()
        return self.labelList[-1]
    
    def sortBySizeDesc(self):
        self.labelList.sort(key=lambda l: l.count, reverse=True)
    
    def getListSortedBySizeDesc(self):
        self.sortBySizeDesc()
        return self.labelList