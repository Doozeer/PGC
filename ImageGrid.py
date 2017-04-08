import numpy as np
from Cell import Cell
from LabelList import LabelList
from CommonUtils import Utils


class ImageGrid(object):
    def __init__(self, image, grid_width, grid_height):
        self.img = image.copy()
        self.width = grid_width
        self.height = grid_height
        temp_grid = [np.hsplit(row, grid_width) for row in np.vsplit(self.img, grid_height)]
        self.grid = [[Cell(rowIdx, colIdx, img) for colIdx, img in enumerate(row)] for rowIdx, row in enumerate(temp_grid)]
        self.labelList = self.label_cells()
    
    def label_cells(self):
        label_list = LabelList()
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                if self.grid[row][col].label is None:
                    cur_label = label_list.get_new_label()
                    self.recursive_label(row, col, cur_label)
        label_list.remove_non_pattern_labels()
        return label_list
    
    def recursive_label(self, row, col, curLabel):
        grid_height = len(self.grid)
        grid_width = len(self.grid[0])
        angle_threshold = 3
        if row in range(grid_height) and col in range(grid_width):
            cur_cell = self.grid[row][col]
            if cur_cell.hasBarcodeFeatures and cur_cell.label is None:
                curLabel.assign_to(cur_cell)
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        new_row = row+i
                        new_col = col+j
                        if new_row in range(grid_height) and new_col in range(grid_width):
                            neighbor = self.grid[new_row][new_col]
                            angle_diff = Utils.calc_angle_diff(cur_cell.orientation, neighbor.orientation)
                            if angle_diff < angle_threshold:
                                self.recursive_label(new_row, new_col, curLabel)

    @staticmethod
    def get_concat_grid(img_grid):
        return np.concatenate([np.concatenate(row, axis=1) for row in img_grid], axis=0)
    
    def get_grid_image(self):
        cell_img_grid = [[cell.img for cell in row]for row in self.grid]
        return ImageGrid.get_concat_grid(cell_img_grid)
    
    def get_label_border_image(self):
        label_img_grid = [[cell.get_label_border_image() for cell in row] for row in self.grid]
        return ImageGrid.get_concat_grid(label_img_grid)
    
    def get_label_orientation_image(self):
        label_img_grid = [[cell.get_label_orientation_image() for cell in row] for row in self.grid]
        return ImageGrid.get_concat_grid(label_img_grid)
    
    def get_label_mask_image(self, label):
        label_mask_grid = [[cell.get_label_mask_image(label) for cell in row] for row in self.grid]
        return ImageGrid.get_concat_grid(label_mask_grid)
