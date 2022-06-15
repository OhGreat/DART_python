from copy import deepcopy
import numpy as np
import astra
from scipy.ndimage import gaussian_filter

class DART():
    def __init__(self, gray_levels, p, rec_shape, 
                proj_geom, projector_id, sinogram):
        """ Instanciate DART with thw following parameters
            Parameters:
                - gray_levels: gray levels known a priori used in the segmentation step.
                - p: probability of a pixel to not be sampled as a free pixel.
                - rec_shape: shape of the volume to create as output.
                - proj_geom: projection geometry to use for the sinogram creation. 
                - projector_id: reference to the astra toolbox projector used to make the projections.
                - sinogram: sinogram as numpy matrix
        """
        self.gray_levels = gray_levels
        # define thresholds for gray levels with start and end values
        self.thresholds = self.update_gray_thresholds()
        self.p = p
        self.c, self.probs = [0,1], [self.p, 1-self.p]
        self.rec_shape = rec_shape
        self.vol_geom = astra.creators.create_vol_geom(self.rec_shape)
        # calculated in advance for efficiency
        self.all_neighbours_idx = [[self.pixel_neighborhood(rec_shape, i,j)
                                    for j in range(self.rec_shape[1])]
                                        for i in range(self.rec_shape[0])]
        self.proj_geom = proj_geom
        self.projector_id = projector_id
        self.sinogram = sinogram
        self.sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)

    def run(self, iters, p=None, gray_levels=None, 
            rec_alg="SART_CUDA", rec_iter=5):
        """ Parameters:
                - iters: (int) number of DART iteration to perform
                - p: (float) probability of a pixel to not be sampled as a free pixel.
                - gray_levels: (list) gray levels known a priori used in the segmentation step.
                - rec_algs: (string) tuple containing the initial and the iterated 
                    reconstruction algorithms to use.
                - rec_iters: (int) number of iterations of the reconstruction subrutine.
            Output:
                (np.array) returns the reconstructed phantom 
                of shape = vol_shape, as a numpy 2D array
        """
        # to run experiments on different gray values
        # and fixed pixel probabilities
        # without reinstanciating DART
        if p is not None:
            self.p = p
        if gray_levels is not None:
            self.gray_levels = gray_levels
            self.thresholds = self.update_gray_thresholds()
        # reconstruction algorithm check
        if rec_alg not in [ "SART", "SART_CUDA",
                            "SIRT", "SIRT_CUDA",
                            "FBP" , "FBP_CUDA"]:
            exit("Select a valid reconstruction algorithm.") 
        # create initial reconstruction
        curr_rec = self.ART(np.full(shape=self.rec_shape,fill_value=0.),
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
            fixed_pixels_idx = np.where(fixed_pixels)
            #create image to feed to reconstructor
            curr_rec[fixed_pixels_idx[0],
                     fixed_pixels_idx[1]] = segmented_img[fixed_pixels_idx[0],
                                                        fixed_pixels_idx[1]]
            # run reconstruction algorithm on free pixels
            curr_rec = self.ART(curr_rec, mask=free_pixels,
                                        alg=rec_alg, iters=rec_iter)
            # smoothing operation except on last iteration
            if i < iters - 1:
                smooth_rec = gaussian_filter(curr_rec, sigma=1)
                curr_rec[free_pixels_idx[0],
                            free_pixels_idx[1]] = smooth_rec[free_pixels_idx[0], 
                                                            free_pixels_idx[1]]
        return curr_rec

    def ART(self, rec, mask=None,
                    alg="SART_CUDA", iters= 5):
        """ Reconstruction with ARM techniques.
            Parameters:
                - rec: initial reconstructed image. (np.array)
                - mask: mask defining the pixels to update, (np.array)
                - alg: name of the reconstruction algorithm to use. (string)
                - iters: number of reconstruction iterations to perform. (int)
            Output:
                - reconstructed image. (np.array)
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
        # free memory
        astra.algorithm.delete(algorithm_id)
        # return the reconstructed values
        return astra.data2d.get(rec_id)

    def update_gray_thresholds(self):
        """ Updates algorithms' thresholds for the currently
            defined gray values.
        """
        return [0] + [(self.gray_levels[i]+self.gray_levels[i+1])/2 
                        for i in range(len(self.gray_levels)-1) ] + [255]

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
                                    if j > -1 and j < img_shape[1] ] # and (x != i & y != j)
        return neighbours

    def boundary_pixels(self, img):
        """ Computes the boundary pixels of the image.
            Returns an image mask where boundary pixels 
            have value 1 and the rest all 0s.

            Parameters:
                - img: define the input image as a numpy array
        """
        # initialize output mask to 0
        bool_mask = np.full(fill_value=False, shape=img.shape[:2], dtype=np.bool8)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                pixel = img[x,y]
                # get actual neighbours values
                neighbours = [img[i] for i in self.all_neighbours_idx[x][y]]
                # update pixel value
                if np.any(neighbours != pixel):
                    bool_mask[x,y] = True
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
                                    p=self.probs).astype(np.bool8)
        return free_pixels

