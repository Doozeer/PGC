from Label import Label


class LabelList(object):
    def __init__(self):
        self.labelList = []
    
    def get_label_count(self):
        return len(self.labelList)
    
    def get_new_label(self):
        new_label = Label(self.get_label_count(), self)
        self.labelList.append(new_label)
        return new_label
    
    def get_cur_label(self):
        if len(self.labelList) < 1:
            self.get_new_label()
        return self.labelList[-1]
    
    def sort_by_size_desc(self):
        self.labelList.sort(key=lambda l: l.count, reverse=True)
    
    def get_list_sorted_by_size_desc(self):
        self.sort_by_size_desc()
        return self.labelList
