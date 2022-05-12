import astra
import numpy as np
from os import mkdir
from os.path import isdir
from PIL import Image


def project_from_2D(phantom_data, n_projections, detector_spacing, apply_noise=False, save_dir=None):
        """ Creates projection for the given input data.
            
            Parameters:
                - phantom_data: numpy array containing the 2D phantom.
                
                - n_projections: defines the number of projections 
                    (number of angles from where to sample measurements 
                
                - detector_spacing: pixel size

                - save_dir: path of the directory to save image representation 
                    of projections, when defined. To be passed as a string.
        """

        img_width, img_height = phantom_data.shape
        # define the number of detectors 
        # as the number of rows in the image
        n_detectors = img_width
        # create angles for measurements
        angles = np.linspace(0, 2*np.pi, n_projections)

        # create astra phantom
        vol_geom = astra.creators.create_vol_geom([img_width,img_height])
        phantom_id = astra.data2d.create('-vol', vol_geom, data=phantom_data)

        # create projections
        proj_geom = astra.create_proj_geom('parallel', detector_spacing, 
                                            n_detectors, angles)
        projector = astra.create_projector('line', proj_geom, vol_geom)
        proj_id, projections = astra.creators.create_sino(phantom_id, projector)

        # TODO: FIX, makes pixels values only 0 or 255
        # Apply Poisson noise.
        if apply_noise:
            projections = np.random.poisson(projections * 10000) / 10000
            projections[projections > 1.1] = 1.1
            projections /= 1.1

        # Save projections if directory has been defined.
        if save_dir != None:
            if not isdir(save_dir):
                mkdir(save_dir)
            projections = np.round(projections * (2**8- 1)).astype(np.uint8)
            for i in range(n_projections):
                if save_dir[-1] != '/':
                    save_dir += '/'
                    Image.fromarray(projections[i]).save(save_dir+f'proj_{i}.png')

        return proj_id, projections