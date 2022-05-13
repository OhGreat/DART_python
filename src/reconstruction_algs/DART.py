import numpy as np

class DART():
    """ gray_levels: R = {p_1, ..,  p_l}
    """

    def __init__(self):
        pass

    def segment(self, img, gray_levels):
        """ Segments the input image to obtain an image with
            only the gray values specified. 
        """
        self.gray_levels = gray_levels
        # defien thresholds for gray levels with start and end values
        self.thresholds = [0] +[(gray_levels[i]+gray_levels[i+1])/2 
                            for i in range(len(gray_levels)-1) ] + [256]
        # Compute segmentation
        for thresh_idx in range(len(self.thresholds)-1):
            cond = (img >= self.thresholds[thresh_idx]) * (img < self.thresholds[thresh_idx+1])
            img[cond] = self.gray_levels[thresh_idx]
        return img

    def pixel_neighborhood(self, img, x, y):
        """ Returns an array containing all the neigh
        """
        # calculate all possible neighbours
        out =[]
        for i in range(x-1,x+2):
            curr_x = np.full(fill_value=i, shape=3)
            curr_y = np.arange(y-1,y+2)
            out.append(np.vstack((curr_x,curr_y)).T)
        out = np.array(out)
    
        max_x, max_y = img.shape
        # remove neighbors with invalid indexes
        out = out[out[..., 0] > 0]
        out = out[out[..., 0] < max_x]
        out = out[out[..., 1] < max_y]
        return out

    def boundary_pixels(self, img):
        """ Computes the boundary pixels of the image.
            Returns an image mask where boundary pixels 
            have value 1 and the rest all 0s.

            Parameters:
                - img: define the input image as a numpy array
        """
        # initialize output mask to 0
        bool_mask = np.full(fill_value=0, shape=img.shape[:2], dtype=np.uint8)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                pixel = img[x,y]
                neighborhood_indexes = self.pixel_neighborhood(img,x,y)
                neighboors = img[tuple(neighborhood_indexes.T)]
                if np.any(neighboors != pixel):
                    bool_mask[x,y] = 1
        return bool_mask


    def non_boundary_free_pixels(self, boundaries, p):
        """ Computes the non boundary free pixels of the image.
            Returns an image mask where non boundary free pixels 
            have value 1 and the rest all 0s.
            
            Parameters:
                - boundaries: binary image mask as a numpy 
                        array that contains the boundary 
                        free pixels to exclude from the free pixels.

                - p: probability that a pixel is not sampled 
                    as a non boundary free pixel
        """

        # we calculate the mask of possible free pixels 
        # and collapse it in 1 dimension
        possible_pixels = np.logical_not(boundaries).reshape(-1)
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
                            shape=boundaries.shape[0]*boundaries.shape[1],
                            dtype=np.uint8)
        for i in range(len(choices)):
            if choices[i] == 1:
                free_out[idxes[i]] = 1

        free_out = free_out.reshape(boundaries.shape)
        return free_out

    def SART(self, W, x):
        """ Simultaneous Algebraic Reconstruction Technique (SART) with
            randomized scheme. Used from DART as the continious update step.
            
            Params:
                - x : reconstructed image 
                - W : projection matrix. Maps the image x to 
                      the vector p of measured data
        """
        pass
