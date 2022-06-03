from copy import deepcopy
from tkinter.messagebox import NO
import numpy as np
import astra
from scipy.ndimage import gaussian_filter

class DART():
    def __init__(self, gray_levels, p, rec_shape, 
                proj_geom, projector_id, sinogram):
        """ Instanciate DART with thw following parameters
        """
        self.gray_levels = gray_levels
        # defien thresholds for gray levels with start and end values
        self.thresholds = [0] +[(gray_levels[i]+gray_levels[i+1])/2 
                            for i in range(len(gray_levels)-1) ] + [255]
        self.p = p
        self.c, self.probs = [0,1], [self.p, 1-self.p]
        self.rec_shape = rec_shape
        self.vol_geom = astra.creators.create_vol_geom(self.rec_shape)
        self.all_neighbours_idx = [[self.pixel_neighborhood(rec_shape, i,j) 
                                for i in range(self.rec_shape[0])]
                                    for j in range(self.rec_shape[1])]
        self.proj_geom = proj_geom
        self.projector_id = projector_id
        self.sinogram = sinogram
        self.sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)

    def run(self, iters, p=None, gray_levels=None, 
            rec_alg="SART_CUDA", rec_iter=5):
        """ Parameters:
                - iters: number of DART iteration to perform
                - gray_levels: gray levels known a priori used in the segmentation step.
                - p: probability of a pixel to not be sampled as a free pixel.
                - vol_shape: shape of the volume to create as output.
                - projector_id: reference to the astra toolbox projector used to make the projections.
                - sino_id: reference to the astra toolbox sinogram.
                - rec_algs: tuple containing the initial and the iterated 
                    reconstruction algorithms to use.
            Output:
                returns the reconstructed phantom 
                of shape = vol_shape, as a numpy 2D array
        """
        # to run experiments on different gray values
        # and fixed pixel probabilities 
        # without reinstanciating DART
        if p is not None:
            self.p = p
        if gray_levels is not None:
            self.gray_levels = gray_levels
        # reconstruction algorithm check
        if rec_alg not in [ "SART", "SART_CUDA",
                            "SIRT", "SIRT_CUDA",
                            "FBP" , "FBP_CUDA"]:
            exit("Select a valid reconstruction algorithm.") 
        # create initial reconstruction
        curr_rec = self.run_rec_alg(np.full(shape=self.rec_shape,fill_value=0.),
                                    mask=None, alg=rec_alg, iters=rec_iter)
        for i in range(iters):
            # segment current reconstructed image
            segmented_img = self.segment(curr_rec)
            # calculate boundary pixels
            boundary_pixels = self.boundary_pixels(segmented_img)
            # calculate free pixels
            free_pixels = self.free_pixels()
            # mask of all free pixels
            free_pixels = np.logical_or(boundary_pixels,free_pixels)
            # take indexes of non fixed pixels
            free_pixels_idx = np.where(free_pixels)
            # fixed pixels
            fixed_pixels = free_pixels == 0
            # take indexes of non fixed pixels
            fixed_pixels_idx = np.where(fixed_pixels > 0)
            #create image to feed to reconstructor
            curr_rec[fixed_pixels_idx[0],
                     fixed_pixels_idx[1]] = segmented_img[fixed_pixels_idx[0],
                                                        fixed_pixels_idx[1]]
            # run reconstruction algorithm on free pixels
            curr_rec = self.run_rec_alg(curr_rec, mask=free_pixels,
                                        alg=rec_alg, iters=rec_iter)
            # smoothing operation except on last iteration
            if i < iters - 1:
                smooth_rec = gaussian_filter(curr_rec, sigma=1)
                curr_rec[free_pixels_idx[0],
                            free_pixels_idx[1]] = smooth_rec[free_pixels_idx[0], 
                                                            free_pixels_idx[1]]
        return curr_rec

    def run_rec_alg(self, rec, mask=None,
                    alg="SART_CUDA", iters= 5):
        """ Reconstruction with ARM techniques.
        """
        if mask is not None:
            # mask free pixels for the fixed pixel sinogram
            free_pixels_idx = np.where(mask > 0)
            fixed_rec = deepcopy(rec)
            fixed_rec[free_pixels_idx[0],
                    free_pixels_idx[1]] = 0
            # create fixed pixels' sinogram
            _, fixed_sino = astra.creators.create_sino(fixed_rec, self.projector_id)
            # create free pixels' sinogram
            free_sino = self.sinogram - fixed_sino
            free_sino_id = astra.data2d.create('-sino', self.proj_geom, free_sino)
            rec_id = astra.data2d.create('-vol', self.vol_geom, data=rec)
        else:  # first reconstrunction
            rec_id = astra.data2d.create('-vol', self.vol_geom, data=0.)
        # define configuration parameters
        alg_cfg = astra.astra_dict(alg)
        if alg_cfg != "SIRT" and alg_cfg != "SIRT_CUDA":
            alg_cfg['ProjectorId'] = self.projector_id
        alg_cfg['ProjectionDataId'] = free_sino_id if mask is not None else self.sinogram_id
        alg_cfg['ReconstructionDataId'] = rec_id
        alg_cfg['option'] = {}
        alg_cfg['option']['MinConstraint'] = 0
        alg_cfg['option']['MaxConstraint'] = 255
        if mask is not None:
            mask_id = astra.data2d.create('-vol', self.vol_geom, mask)
            alg_cfg['option']['ReconstructionMaskId'] = mask_id
        #define algorithm
        algorithm_id = astra.algorithm.create(alg_cfg)
        # run the algorithm
        astra.algorithm.run(algorithm_id, iters)
        # return the reconstructed values
        return astra.data2d.get(rec_id)

    def segment(self, img):
        """ Segments the input image to obtain an image with
            only the gray values specified. 
        """
        # compute segmentation
        segmented_img = np.full(img.shape, 0, dtype=np.uint8)
        for thresh_idx in range(len(self.thresholds)-1):
            cond = (img >= self.thresholds[thresh_idx]) * (img <= self.thresholds[thresh_idx+1])
            segmented_img[cond] = self.gray_levels[thresh_idx]
        return segmented_img

    def pixel_neighborhood(self, img_shape, x, y):
        """ Returns an array containing all the neighbours of the given pixel
        """
        # calculate all possible neighbours
        # related to the x,y coordinates
        neighbours = [(i,j) 
                        for i in range(x-1, x+2) 
                            if i > -1 and i < img_shape[0]
                                for j in range(y-1, y+2) 
                                    if j > -1 and j < img_shape[1] ]
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
                # get actual neighbours values
                neighbours = [img[i] for i in self.all_neighbours_idx[x][y]]
                # update pixel value
                if np.any(neighbours != pixel):
                    bool_mask[x,y] = 1
        return bool_mask

    def free_pixels(self):
        """ Computes the free pixels of the image.
            
            Parameters:
                - rec_shape: shape of the image for which 
                    to calculate the free pixels 

                - p: probability that a pixel is not sampled 
                    as a non boundary free pixel
        """
        
        free_pixels = np.random.choice(a=self.c, 
                                    size=self.rec_shape, 
                                    p=self.probs).astype(np.uint8)
        return free_pixels

def SART(vol_geom, vol_data, projector_id, sino_id, iters=2000, use_gpu=False):
        """ Simultaneous Algebraic Reconstruction Technique (SART) with
            randomized scheme. Used from DART as the continious update step.
        """
        # create starting reconstruction
        rec_id = astra.data2d.create('-vol', vol_geom, data=vol_data)
        # define SART configuration parameters
        alg_cfg = astra.astra_dict('SART_CUDA' if use_gpu else 'SART')
        alg_cfg['ProjectorId'] = projector_id
        alg_cfg['ProjectionDataId'] = sino_id
        alg_cfg['ReconstructionDataId'] = rec_id
        alg_cfg['option'] = {}
        alg_cfg['option']['MinConstraint'] = 0
        alg_cfg['option']['MaxConstraint'] = 255
        # define algorithm
        algorithm_id = astra.algorithm.create(alg_cfg)
        # run the algirithm
        astra.algorithm.run(algorithm_id, iters)
        # create reconstruction data
        rec = astra.data2d.get(rec_id)

        return rec_id, rec

def SIRT(vol_geom, vol_data, sino_id, iters=2000, use_gpu=False):
        # create starting reconstruction
        rec_id = astra.data2d.create('-vol', vol_geom, data=vol_data)
        # define SIRT config params
        alg_cfg = astra.astra_dict('SIRT_CUDA' if use_gpu else 'SIRT')
        alg_cfg['ProjectionDataId'] = sino_id
        alg_cfg['ReconstructionDataId'] = rec_id
        alg_cfg['option'] = {}
        alg_cfg['option']['MinConstraint'] = 0
        alg_cfg['option']['MaxConstraint'] = 255
        # define algorithm
        alg_id = astra.algorithm.create(alg_cfg)
        # run the algorithm
        astra.algorithm.run(alg_id, iters)
        # create reconstruction data
        rec = astra.data2d.get(rec_id)

        return rec_id, rec

def FBP(vol_geom, vol_data, projector_id, sino_id, iters=2000, use_gpu=False):
    # create starting reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom, data=vol_data)
    # define SART configuration parameters
    alg_cfg = astra.astra_dict('FBP_CUDA' if use_gpu else 'FBP')
    alg_cfg['ProjectorId'] = projector_id
    alg_cfg['ProjectionDataId'] = sino_id
    alg_cfg['ReconstructionDataId'] = rec_id
    alg_cfg['option'] = {}
    alg_cfg['option']['MinConstraint'] = 0
    alg_cfg['option']['MaxConstraint'] = 255
    # define algorithm
    algorithm_id = astra.algorithm.create(alg_cfg)
    # run the algirithm
    astra.algorithm.run(algorithm_id, iters)
    # create reconstruction data
    rec = astra.data2d.get(rec_id)

    return rec_id, rec