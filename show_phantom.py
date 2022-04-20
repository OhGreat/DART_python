import foam_ct_phantom
import h5py
from PIL import Image

phantom_path = "data/foam/phantoms/test_phantom.h5"
volume_path = "data/foam/geometries/test_phantom_vol.h5"


phantom = foam_ct_phantom.FoamPhantom(phantom_path)

geom = foam_ct_phantom.VolumeGeometry(256,256,1,3/256)

phantom.generate_volume(volume_path, geom)

vol = foam_ct_phantom.load_volume(volume_path)

img = Image.fromarray(vol[0])
img.show()