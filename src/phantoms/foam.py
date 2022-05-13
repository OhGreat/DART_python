import foam_ct_phantom
from PIL import Image
from os import makedirs
from os.path import exists

def create_foam(filename, n_spheres = 1000, seed=0):
    """ Creates a foam like phantom image with volume
        and saves them in the data/foam/ directory
    """

    # define save paths
    if not exists('data/foam'):
        makedirs('data/foam')
    phantom_save_path = 'data/foam/'+ filename + '_phantom.h5'
    volume_save_path = "data/foam/" + filename + '_volume.h5'
    img_save_path = "data/foam/" + filename + '_img.png'
    # generate phantom
    foam_ct_phantom.FoamPhantom.generate(phantom_save_path,seed,
                                        nspheres_per_unit=n_spheres)
    # load generated phantom
    phantom = foam_ct_phantom.FoamPhantom(phantom_save_path)
    # generate phantom geometry
    geom = foam_ct_phantom.VolumeGeometry(256,256,1,3/256)
    # generate phantom volume from phantom geometry
    phantom.generate_volume(volume_save_path, geom)
    # load volume
    vol = foam_ct_phantom.load_volume(volume_save_path)
    #save image
    img = Image.fromarray(vol[0]*255)
    img = img.convert('L')
    img.save(img_save_path)