import random as rnd


class Label(object):
    def __init__(self, label_number, parent_list):
        self.count = 0
        self.number = label_number
        self.parentList = parent_list
        self.color = (rnd.randint(0,255), rnd.randint(0, 255), rnd.randint(0, 255))
        self.orientationSum = 0.0
    
    def assign_to(self, obj):
        if obj.label is not None:
            raise ValueError('Trying to assign new Label to an object that already has a label.')
        obj.label = self
        self.count += 1
        self.orientationSum += obj.orientation

    def get_mean_orientation(self):
        if self.count != 0:
            return self.orientationSum / self.count
        else:
            return 0.0
