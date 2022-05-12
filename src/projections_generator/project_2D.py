import astra
import numpy as np

def project_from_2D(phantom_data, n_projections, detector_spacing):
        """ Creates projection for the given input data.
            
            Parameters:
                - phantom_data: numpy array containing the 2D phantom.
                
                - n_projections: defines the number of projections 
                    (number of angles from where to sample measurements 
                
                - detector_spacing: pixel size
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

        return proj_id, projections