import astra
import numpy as np
from os import mkdir
from os.path import isdir
from PIL import Image

def project_from_2D(phantom_id, vol_geom, n_projections, 
                    n_detectors, detector_spacing, angles, 
                    noise_factor=None, save_dir=None, use_gpu=False):
        """ Creates projection for the given input data.
            
            Parameters:
                - phantom_data: numpy array containing the 2D phantom.
                - n_projections: defines the number of projections 
                    (number of angles from where to sample measurements 
                - detector_spacing: pixel size
                - angles:
                - noise_factor:
                - save_dir: path of the directory to save image representation 
                    of projections, when defined. To be passed as a string.
                - use_gpu: (boolean) set to True to use gpu.

            Returns:
                projector_id, sinogram_id and sinogram matrix
        """

        # create projection geometry
        proj_geom = astra.create_proj_geom('parallel', detector_spacing, 
                                            n_detectors, angles)
        # choose projector
        if use_gpu:
            proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
        else:
            proj_id = astra.create_projector('linear', proj_geom, vol_geom)
        sino_id, sinogram = astra.creators.create_sino(phantom_id, proj_id)
        # Apply Poisson noise.
        if noise_factor != None:
            sinogram += np.random.poisson(lam=noise_factor, size=sinogram.shape)
            sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
        # Save projections as images, if directory has been defined.
        if save_dir != None:
            if save_dir[-1] != '/':
                save_dir += '/'
            if not isdir(save_dir):
                mkdir(save_dir)
            proj_for_img = np.round(sinogram * (2**8- 1)).astype(np.uint8)
            for i in range(n_projections):
                Image.fromarray(proj_for_img[i]).save(save_dir+f'proj_{i}.png')

        return proj_id, sino_id, sinogram