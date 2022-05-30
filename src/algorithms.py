import numpy as np
import astra
from scipy.ndimage import gaussian_filter

class DART():

    def __init__(self):
        pass

    def original(self, iters, gray_levels, p, 
                vol_shape, projector_id, sino_id, 
                rec_algs=("SIRT","FBP"), rec_iter=200, use_gpu=False):
        """ Original implementation as in the paper
            Parameters:
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
        # create volume geometry
        vol_geom = astra.creators.create_vol_geom(vol_shape)
        # initialize current reconstruction
        if rec_algs[0] == "SART":
            _, curr_reconstr = self.SART(vol_geom, 0, projector_id, 
                                    sino_id, iters=rec_iter,
                                    use_gpu=use_gpu)
        elif rec_algs[0] == "SIRT":
            _, curr_reconstr = self.SIRT(vol_geom, 0, sino_id, 
                                        iters=rec_iter,
                                        use_gpu=use_gpu)
        elif rec_algs[0] == "FBP":
            _, curr_reconstr = self.FBP(vol_geom, 0, projector_id, 
                                        sino_id, iters=rec_iter,
                                        use_gpu=use_gpu)
        else:
            exit("Please choose valid reconstruction algorithms.")

        for i in range(iters):
            # segment current reconstructed image
            segmented_img = self.segment(curr_reconstr, gray_levels)
            # calculate boundary pixels
            boundary_pixels = self.boundary_pixels(segmented_img)
            # calculate free pixels
            free_pixels = self.free_pixels(vol_shape,p)
            # mask used to set the pixels from the new reconstruction
            non_fixed_pixels = np.logical_or(boundary_pixels,free_pixels)
            # take indexes of free pixels (non fixed pixels)
            free_pixels_idx = np.where(non_fixed_pixels)
            # create image to feed to reconstructor
            segmented_img[free_pixels_idx[0], 
                        free_pixels_idx[1]] = curr_reconstr[free_pixels_idx[0],
                                                            free_pixels_idx[1]]
            # calculate new reconstruction
            if rec_algs[1] == "FBP":
                _, new_reconstr = self.FBP(vol_geom, segmented_img,
                                            projector_id, sino_id,
                                            iters=rec_iter, use_gpu=use_gpu)
            elif rec_algs[1] == "SART":
                _, new_reconstr = self.SART(vol_geom, segmented_img, 
                                            projector_id, sino_id,
                                            iters=rec_iter, use_gpu=use_gpu)
            elif rec_algs[1] == "SIRT":
                _, new_reconstr = self.SIRT(vol_geom, segmented_img, sino_id, 
                                            iters=rec_iter, use_gpu=use_gpu)
            else:
                exit("Please choose valid reconstruction algorithms.")
            # update the current free pixels
            curr_reconstr[free_pixels_idx[0], 
                        free_pixels_idx[1]] = new_reconstr[free_pixels_idx[0], 
                                                            free_pixels_idx[1]]
            # smoothing operation on free pixels
            if i < iters - 1: 
                smooth_rec = gaussian_filter(curr_reconstr, sigma=1)
                curr_reconstr[free_pixels_idx[0], 
                            free_pixels_idx[1]] = smooth_rec[free_pixels_idx[0], 
                                                            free_pixels_idx[1]]
        return curr_reconstr

    def __call__(self, iters, gray_levels, p, 
                vol_shape, projector_id, sino_id,
                rec_iter=200, use_gpu=False):
        """ New implementation with one less step and better reconstruction.
            Parameters:
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
        # create volume geometry
        vol_geom = astra.creators.create_vol_geom(vol_shape)
        # create initial reconstruction
        _, curr_reconstr = self.SART(vol_geom, 0, projector_id, 
                                    sino_id, iters=rec_iter,
                                    use_gpu=use_gpu)
        for i in range(iters):
            # segment current reconstructed image
            curr_reconstr = self.segment(curr_reconstr, gray_levels)
            # calculate new reconstruction
            _, new_reconstr = self.SART(vol_geom, curr_reconstr, 
                                            projector_id, sino_id,
                                            iters=rec_iter, use_gpu=use_gpu)
            # calculate boundary pixels
            boundary_pixels = self.boundary_pixels(curr_reconstr)
            # calculate free pixels
            free_pixels = self.free_pixels(vol_shape,p)
            # mask used to set the pixels from the new reconstruction
            non_fixed_pixels = np.logical_or(boundary_pixels,free_pixels)
            # take indexes of non fixed pixels
            free_pixels_idx = np.where(non_fixed_pixels)
            # update the current free pixels
            curr_reconstr[free_pixels_idx[0], 
                        free_pixels_idx[1]] = new_reconstr[free_pixels_idx[0], 
                                                            free_pixels_idx[1]]
            # smoothing operation except on for last iteration
            if i < iters - 1:
                smooth_rec = gaussian_filter(curr_reconstr, sigma=1)
                curr_reconstr[free_pixels_idx[0], 
                            free_pixels_idx[1]] = smooth_rec[free_pixels_idx[0], 
                                                            free_pixels_idx[1]]
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

    def SART(self, vol_geom, vol_data, projector_id, sino_id, iters=2000, use_gpu=False):
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

    def SIRT(self, vol_geom, vol_data, sino_id, iters=2000, use_gpu=False):
        # create starting reconstruction
        rec_id = astra.data2d.create('-vol', vol_geom, data=vol_data)
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

    def FBP(self, vol_geom, vol_data, projector_id, sino_id, iters=2000, use_gpu=False):
        # create starting reconstruction
        rec_id = astra.data2d.create('-vol', vol_geom, data=vol_data)
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