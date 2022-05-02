import numpy as np

class DART():
    """ gray_levels: R = {p_1, ..,  p_l}
    """

    def __init__(self, gray_levels, fix_p = 0.5):
        self.gray_levels = gray_levels
        # defien thresholds for gray levels with start and end values
        self.thresholds = [0] +[(gray_levels[i]+gray_levels[i+1])/2 
                            for i in range(len(gray_levels)-1) ] + [256]
        self.fix_p = fix_p

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
    
        # remove neighbors with invalid indexes
        out = out[out[..., 0] >= 0]
        out = out[out[..., 0] <= 255]
        out = out[out[..., 1] <= 255]
        return out

    def boundary_pixels(self, img):
        """ Computes the boundary pixels of the image.
            Returns an image mask where boundary pixels have value 1 and the rest all 0s.
        """
        # initialize output mask to 0
        bool_mask = np.full(fill_value=0, shape=img.shape[:2], dtype=np.uint8)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                pixel = img[x,y]
                neighborhood_indexes = self.pixel_neighborhood(x,y)
                neighboors = img[tuple(neighborhood_indexes.T)]
                if np.any(neighboors != pixel):
                    bool_mask[x,y] = 1
        return bool_mask


    def non_boundary_free_pixels(self,mask, p):
        """ Computes the non boundary free pixels of the image.
            Returns an image mask where non boundary free pixels have value 1 and the rest all 0s.
        """

        # we calculate the mask of possible free pixels 
        # and collapse it in 1 dimension
        possible_pixels = np.logical_not(mask).reshape(-1)
        # get indexes of all available pixels
        idxes = np.hstack(np.nonzero(possible_pixels))

        # we sample randomly with probability 1-p the free pixels
        # by assigning a binary variable to each available idx.
        c = [0,1]
        probs = [p, 1-p]
        choices = np.random.choice(a=c, size=idxes.shape, p=probs)

        # we create a new output mask of the size of the image
        # and substitute ones in the indexes from 'idxes' where 
        # the boolean flags are True
        free_out = np.full(fill_value=0,
                            shape=mask.shape[0]*mask.shape[1],
                            dtype=np.uint8)
        for i in range(len(choices)):
            if choices[i] == 1:
                free_out[idxes[i]] = 1

        free_out = free_out.reshape(mask.shape)
        return free_out