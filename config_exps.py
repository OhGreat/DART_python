from itertools import count
import astra
import numpy as np
from PIL import Image
from src.algorithms import DART
from src.projections import project_from_2D
from os import listdir, makedirs
from os.path import exists

def main():
    # total iterations
    iters = 2000
    # define sample interval
    error_intervals = 50
    # exp range
    exp_range = list(range(0,iters+1,error_intervals))
    exp_range = [10, 20, 30] + exp_range
    # save image intervals
    img_intervals = [10, 20, 30, 50, 500, 1000, 2000]
    # phantom family
    phantoms = "semilunars"
    # input directory
    in_dir = f"phantoms/{phantoms}/"

    curr_phantom_idx = 0
    for filename in sorted(listdir(in_dir)):
        print("curr phantom:", curr_phantom_idx)
        phantom = np.array(Image.open(in_dir+filename), dtype=np.uint8)
        img_width, img_height = phantom.shape

        # experiments with 50 projections (angles), 512 detectors 
        # and 1 detector spacing
        n_proj, n_detectors, det_spacing = 50, 512, 1
        vol_geom = astra.creators.create_vol_geom([img_width,img_height])
        phantom_id = astra.data2d.create('-vol', vol_geom, data=phantom)
        projector_id, sino_id, sinogram = project_from_2D(phantom_id, 
                                                        vol_geom,n_proj,
                                                        n_detectors,
                                                        det_spacing,
                                                        use_gpu=True)
        # SART
        errors = []
        out_dir = f"results/SART_{n_proj}proj_{n_detectors}det_{det_spacing}sp/{phantoms}/"
        if not exists(out_dir):
            makedirs(out_dir)
        exp_name = 'sart_' + str(curr_phantom_idx)
        for curr_iters in exp_range:
            _, sart_res = DART().SART(vol_geom, projector_id, sino_id, curr_iters, use_gpu=True)
            sart_res[sart_res > 255] = 255
            sart_res[sart_res < 0] = 0
            errors.append(np.abs(phantom - sart_res).mean())
            print("curr err:", np.abs(phantom - sart_res).mean())
            if curr_iters in img_intervals:
                img = Image.fromarray(sart_res.astype(np.uint8))
                img.save(out_dir+exp_name+"_"+str(curr_iters)+".png")
                print(out_dir+exp_name+"_"+str(curr_iters)+".png")
        np.save(out_dir+exp_name, errors)

        # SIRT
        errors = []
        out_dir = f"results/SIRT_{n_proj}proj_{n_detectors}det_{det_spacing}sp/{phantoms}/"
        if not exists(out_dir):
            makedirs(out_dir)
        exp_name = 'sirt_' + str(curr_phantom_idx)
        for curr_iters in exp_range:
            _, sirt_res = DART().SIRT(vol_geom, sino_id, curr_iters, use_gpu=True)
            sirt_res[sirt_res > 255] = 255
            sirt_res[sirt_res < 0] = 0
            errors.append(np.abs(phantom - sirt_res).mean())
            print("curr err:", np.abs(phantom - sirt_res).mean())
            if curr_iters in img_intervals: 
                img = Image.fromarray(sirt_res.astype(np.uint8))
                img.save(out_dir+exp_name+"_"+str(curr_iters)+".png")
                print(np.abs(phantom - sirt_res).mean(),out_dir+exp_name+"_"+str(curr_iters)+".png")
        np.save(out_dir+exp_name, errors)

        # FBP
        errors = []
        out_dir = f"results/FBP_{n_proj}proj_{n_detectors}det_{det_spacing}sp/{phantoms}/"
        if not exists(out_dir):
            makedirs(out_dir)
        exp_name = 'fbp_' + str(curr_phantom_idx)
        for curr_iters in exp_range:
            _, fbp_res = DART().FBP(vol_geom, projector_id, sino_id, curr_iters, use_gpu=True)
            fbp_res[fbp_res > 255] = 255
            fbp_res[fbp_res < 0] = 0
            errors.append(np.abs(phantom - fbp_res).mean())
            print("curr err:", np.abs(phantom - fbp_res).mean())
            if curr_iters in img_intervals: 
                img = Image.fromarray(fbp_res.astype(np.uint8))
                img.save(out_dir+exp_name+"_"+str(curr_iters)+".png")
                print(np.abs(phantom - fbp_res).mean(),out_dir+exp_name+"_"+str(curr_iters)+".png")
        np.save(out_dir+exp_name, errors)
        
        curr_phantom_idx += 1

if __name__ == "__main__":
    main()