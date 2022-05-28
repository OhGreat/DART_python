from itertools import count
from typing import ItemsView
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
    # phantom family
    phantoms = "semilunars"
    # input directory
    in_dir = f"phantoms/{phantoms}/"

    # output directory
    out_dir_proj = "results/n_proj/"
    out_dir_angles = "results/angle_range/"
    if not exists(out_dir_proj):
            makedirs(out_dir_proj)
    if not exists(out_dir_angles):
            makedirs(out_dir_angles)

    # define number of projections and angles
    n_projections = range(10, 120, 10)
    angle_range = [x/20 for x in range(1,21)]
    
    # choose a phantom
    phantom_name = sorted(listdir(in_dir))[0]
    phantom = np.array(Image.open(in_dir+phantom_name), dtype=np.uint8)
    img_width, img_height = phantom.shape

    # Number of projections experiments
    proj_errors_sart = []
    proj_errors_sirt = []
    proj_errors_rbf = []
    proj_errors_dart = []
    for curr_proj in n_projections:
        print("curr_proj:", curr_proj)
        n_proj, n_detectors, det_spacing = curr_proj, 512, 1
        vol_geom = astra.creators.create_vol_geom([img_width,img_height])
        phantom_id = astra.data2d.create('-vol', vol_geom, data=phantom)
        projector_id, sino_id, _ = project_from_2D(phantom_id, 
                                                        vol_geom,n_proj,
                                                        n_detectors,
                                                        det_spacing,
                                                        ang_mul=1,
                                                        use_gpu=True)
        # SART
        _, sart_res = DART().SART(vol_geom, projector_id, sino_id, iters, use_gpu=True)
        sart_res[sart_res > 255] = 255
        sart_res[sart_res < 0] = 0
        proj_errors_sart.append(np.abs(phantom - sart_res).mean())

        # SIRT
        _, sirt_res = DART().SIRT(vol_geom, sino_id, iters, use_gpu=True)
        sirt_res[sirt_res > 255] = 255
        sirt_res[sirt_res < 0] = 0
        proj_errors_sirt.append(np.abs(phantom - sirt_res).mean())

        # RBF
        _, fbp_res = DART().FBP(vol_geom, projector_id, sino_id, iters, use_gpu=True)
        fbp_res[fbp_res > 255] = 255
        fbp_res[fbp_res < 0] = 0
        proj_errors_rbf.append(np.abs(phantom - fbp_res).mean())

        # DART
        dart = DART()
        gray_lvls = np.unique(phantom).astype(np.float32) 
        dart_res = dart(iters=50,
                    gray_levels=gray_lvls,p=0.85,
                    vol_shape=phantom.shape,
                    projector_id=projector_id, sino_id=sino_id,
                    SART_iter=200, use_gpu=True)
        proj_errors_dart.append(dart_res)

    np.save(out_dir_proj+"SART", proj_errors_sart)
    np.save(out_dir_proj+"SIRT", proj_errors_sirt)
    np.save(out_dir_proj+"RBF", proj_errors_rbf)
    np.save(out_dir_proj+"DART", proj_errors_dart)
    print()
    
    # Number of angles experiments
    ang_errors_sart = []
    ang_errors_sirt = []
    ang_errors_rbf = []
    ang_errors_dart = []
    for curr_ang in angle_range:
        print("curr_ang mul:", curr_ang)
        n_proj, n_detectors, det_spacing = 50, 512, 1
        vol_geom = astra.creators.create_vol_geom([img_width,img_height])
        phantom_id = astra.data2d.create('-vol', vol_geom, data=phantom)
        projector_id, sino_id, _ = project_from_2D(phantom_id, 
                                                        vol_geom,n_proj,
                                                        n_detectors,
                                                        det_spacing,
                                                        ang_mul=curr_ang,
                                                        use_gpu=True)
        # SART
        _, sart_res = DART().SART(vol_geom, projector_id, sino_id, iters, use_gpu=True)
        sart_res[sart_res > 255] = 255
        sart_res[sart_res < 0] = 0
        ang_errors_sart.append(np.abs(phantom - sart_res).mean())

        # SIRT
        _, sirt_res = DART().SIRT(vol_geom, sino_id, iters, use_gpu=True)
        sirt_res[sirt_res > 255] = 255
        sirt_res[sirt_res < 0] = 0
        ang_errors_sirt.append(np.abs(phantom - sirt_res).mean())

        # RBF
        _, fbp_res = DART().FBP(vol_geom, projector_id, sino_id, iters, use_gpu=True)
        fbp_res[fbp_res > 255] = 255
        fbp_res[fbp_res < 0] = 0
        ang_errors_rbf.append(np.abs(phantom - fbp_res).mean())

        # DART
        dart = DART()
        gray_lvls = np.unique(phantom).astype(np.float32) 
        dart_res = dart(iters=50,
                    gray_levels=gray_lvls,p=0.85,
                    vol_shape=phantom.shape,
                    projector_id=projector_id, sino_id=sino_id,
                    SART_iter=200, use_gpu=True)
        ang_errors_dart.append(dart_res)


    np.save(out_dir_angles+"SART", ang_errors_sart)
    np.save(out_dir_angles+"SIRT", ang_errors_sirt)
    np.save(out_dir_angles+"RBF", ang_errors_rbf)
    np.save(out_dir_angles+"DART", ang_errors_dart)


if __name__ == "__main__":
    main()