import numpy as np
import astra
from scipy.ndimage import gaussian_filter
from os import makedirs
from os.path import exists

class DART():

    def __init__(self):
        pass

    def __call__(self, iters, gray_levels, p, 
                vol_shape, projector_id, sino_id, 
                SART_iter, use_gpu=False,
                stats=None):
        """ TODO: add documentation
            Parameters:
            - stats: tuple of 3 values. (every_n, phantom, save_path)
        """
    
        # check directory exists
        if stats != None:
            if not exists(stats[2]):
                makedirs(stats[2])
            

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
                                        sino_id, iters=SART_iter,
                                        use_gpu=use_gpu)
            # take indexes of non fixed pixels
            free_pixels_idx = np.where(non_fixed_pixels)
            # update the current free pixels
            curr_reconstr[free_pixels_idx[0], 
                        free_pixels_idx[1]] = new_reconstr[free_pixels_idx[0], 
                                                            free_pixels_idx[1]]
            # smoothing operation
            smooth_rec = gaussian_filter(curr_reconstr, sigma=1)
            curr_reconstr[free_pixels_idx[0], 
                        free_pixels_idx[1]] = smooth_rec[free_pixels_idx[0], 
                                                        free_pixels_idx[1]]
            # save statistics if stats_dir is defined

        return curr_reconstr

    def segment(self, img, gray_levels):
        """ Segments the input image to obtain an image with
            only the gray values specified. 
        """
        self.gray_levels = gray_levels
        # defien thresholds for gray levels with start and end values
        self.thresholds = [0] +[(gray_levels[i]+gray_levels[i+1])/2 
                            for i in range(len(gray_levels)-1) ] + [255]
        # Compute segmentation
        for thresh_idx in range(len(self.thresholds)-1):
            cond = (img >= self.thresholds[thresh_idx]) * (img < self.thresholds[thresh_idx+1])
            img[cond] = self.gray_levels[thresh_idx]
        return img

    def pixel_neighborhood(self, img, x, y):
        """ Returns an array containing all the neighbours of the given pixel
        """
        # calculate all possible neighbours
        # related to the x,y coordinates
        max_x, max_y = img.shape
        neighbours = [(i,j) 
                        for i in range(x-1, x+2) 
                            if i > -1 and i < max_x
                                for j in range(y-1, y+2) 
                                    if j > -1 and j < max_y ]
        return neighbours

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
                # get curr pixel neighbours indexes
                neigh_idxes = self.pixel_neighborhood(img,x,y)
                # get actual neighbours values
                neighbours = [img[i] for i in neigh_idxes]
                # update pixel value
                if np.any(neighbours != pixel):
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
        c, probs = [0,1], [p, 1-p]
        free_pixels = np.random.choice(a=c, 
                                    size=img_shape, 
                                    p=probs).astype(np.uint8)
        return free_pixels

    def SART(self, vol_geom, projector_id, sino_id, iters=2000, use_gpu=False):
        """ Simultaneous Algebraic Reconstruction Technique (SART) with
            randomized scheme. Used from DART as the continious update step.
        """
        # create empty reconstruction
        rec_id = astra.data2d.create('-vol', vol_geom, data=0)
        # define SART configuration parameters
        alg_cfg = astra.astra_dict('SART_CUDA' if use_gpu else 'SART')
        alg_cfg['ProjectorId'] = projector_id
        alg_cfg['ProjectionDataId'] = sino_id
        alg_cfg['ReconstructionDataId'] = rec_id
        # define algorithm
        algorithm_id = astra.algorithm.create(alg_cfg)
        # run the algirithm
        astra.algorithm.run(algorithm_id, iters)
        # create reconstruction data
        rec = astra.data2d.get(rec_id)
        # constraint the max/min values
        rec[rec > 255] = 255
        rec[rec < 0] = 0

        return rec_id, rec

    def SIRT(self, vol_geom, sino_id, iters=2000, use_gpu=False):
        # create empty reconstruction
        rec_id = astra.data2d.create('-vol', vol_geom, data=0)
        # define SIRT config params
        alg_cfg = astra.astra_dict('SIRT_CUDA' if use_gpu else 'SIRT')
        alg_cfg['ProjectionDataId'] = sino_id
        alg_cfg['ReconstructionDataId'] = rec_id
        # define algorithm
        alg_id = astra.algorithm.create(alg_cfg)
        # run the algorithm
        astra.algorithm.run(alg_id, iters)
        # create reconstruction data
        rec = astra.data2d.get(rec_id)
        # constraint min/max values
        rec[rec > 255] = 255
        rec[rec < 0] = 0

        return rec_id, rec

    def FBP(self, vol_geom, projector_id, sino_id, iters=2000, use_gpu=False):
        """ Simultaneous Algebraic Reconstruction Technique (SART) with
            randomized scheme. Used from DART as the continious update step.
        """
        # create empty reconstruction
        rec_id = astra.data2d.create('-vol', vol_geom, data=0)
        # define SART configuration parameters
        alg_cfg = astra.astra_dict('FBP_CUDA' if use_gpu else 'FBP')
        alg_cfg['ProjectorId'] = projector_id
        alg_cfg['ProjectionDataId'] = sino_id
        alg_cfg['ReconstructionDataId'] = rec_id

        # define algorithm
        algorithm_id = astra.algorithm.create(alg_cfg)
        # run the algirithm
        astra.algorithm.run(algorithm_id, iters)
        # create reconstruction data
        rec = astra.data2d.get(rec_id)
        # constraint the max/min values
        rec[rec > 255] = 255
        rec[rec < 0] = 0

        return rec_id, rec