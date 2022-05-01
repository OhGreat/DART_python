import numpy as np

class DART():
    """ gray_levels: R = {ρ_1, ..,  ρ_l}
    """

    def __init__(self, gray_levels):
        self.gray_levels = gray_levels
        # defien thresholds for gray levels with start and end values
        self.thresholds = [0] +[(gray_levels[i]+gray_levels[i+1])/2 
                            for i in range(len(gray_levels)-1) ] + [255]

    def segment(self, img):
        """ Segments the input image to obtain an image with
            only the gray values specified. 
        """
        for thresh_idx in range(len(self.thresholds)-1):
            cond = (img >= self.thresholds[thresh_idx]) * (img < self.thresholds[thresh_idx+1])
            img[cond] = self.gray_levels[thresh_idx]
        return img


    def pixel_neighborhood(self, x, y):
        """ Returns an array containing all the 
        """
        # calculate all possible neighbours
        out =[]
        for i in range(x-1,x+2):
            curr_x = np.full(fill_value=i, shape=3)
            curr_y = np.arange(y-1,y+2)
            out.append(np.vstack((curr_x,curr_y)).T)
        out = np.array(out)
    
        # remove neighbors with negative indexes
        out = out[out[..., 0] >= 0]
        # remove neighbors with values > 255
        out = out[out[..., 0] <= 255]
        out = out[out[..., 1] <= 255]
        return out

    def boundary_pixels(self, img):
        """ Computes the boundary pixels of the image.
        """
        # initialize output mask to 0
        bool_mask = np.full(fill_value=0, shape=img.shape[:2])
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                pixel = img[x,y]
                neighborhood_indexes = self.pixel_neighborhood(x,y)
                neighboors = img[tuple(neighborhood_indexes.T)]
                if np.any(neighboors != pixel):
                    bool_mask[x,y] = 1
        return bool_mask