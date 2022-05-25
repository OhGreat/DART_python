import numpy as np
import astra

class DART():

    def __init__(self):
        pass

    def __call__(self, iters, gray_levels, p, 
                vol_shape, projector_id, sino_id, SART_iter):
        """ TODO: add documentation
        """
        # create volume geometry
        vol_geom = astra.creators.create_vol_geom(vol_shape)
        # initialize current reconstruction
        _, curr_reconstr = self.SART(vol_geom,projector_id, 
                                    sino_id, 
                                    iters=SART_iter)

        for i in range(iters):
            # segment current reconstructed image
            segmented_img = self.segment(curr_reconstr, gray_levels)
            # calculate boundary pixels
            boundary_pixels = self.boundary_pixels(segmented_img)
            # calculate free pixels
            free_pixels = self.free_pixels(vol_shape,p)
            # mask used to set the pixels from the new reconstruction
            non_fixed_pixels = np.logical_or(boundary_pixels,free_pixels)
            # calculate new SART reconstruction
            _, new_reconstr = self.SART(vol_geom, projector_id, 
                                        sino_id, iters=SART_iter)
            # take indexes of non fixed pixels
            free_pixels_idx = np.where(non_fixed_pixels)
            # update the current free pixels
            curr_reconstr[free_pixels_idx[0], 
                        free_pixels_idx[1]] = new_reconstr[free_pixels_idx[0], 
                                                            free_pixels_idx[1]]
        return curr_reconstr

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
        """ Returns an array containing all the neighbours of the given pixel
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

    def free_pixels(self, img_shape, p):
        """ Computes the free pixels of the image.
            
            Parameters:
                - img_shape: shape of the image for which 
                    to calculate the free pixels 

                - p: probability that a pixel is not sampled 
                    as a non boundary free pixel
        """
        c = [0,1]
        probs = [p, 1-p]
        free_pixels = np.random.choice(a=c, 
                                    size=img_shape, 
                                    p=probs).astype(np.uint8)
        return free_pixels

    def SART(self, vol_geom, projector_id, sino_id, iters=200):
        """ Simultaneous Algebraic Reconstruction Technique (SART) with
            randomized scheme. Used from DART as the continious update step.
        """
        # create empty reconstruction
        reconstruction_id = astra.data2d.create('-vol', vol_geom, data=0)
        # define SART configuration parameters
        alg_cfg = astra.astra_dict('SART')
        alg_cfg['ProjectorId'] = projector_id
        alg_cfg['ProjectionDataId'] = sino_id
        alg_cfg['ReconstructionDataId'] = reconstruction_id
        alg_cfg['MinConstraint'] = 0
        alg_cfg['MaxConstraint'] = 255
        alg_cfg['ProjectionOrder'] = 'random'  # is set as default

        # define algorithm
        algorithm_id = astra.algorithm.create(alg_cfg)
        # run the algirithm
        astra.algorithm.run(algorithm_id, iters)
        # create reconstruction data
        reconstruction = astra.data2d.get(reconstruction_id)

        reconstruction[reconstruction > 255] = 255
        reconstruction[reconstruction < 0] = 0

        return reconstruction_id, reconstruction
