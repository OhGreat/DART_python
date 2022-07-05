import astra
import numpy as np
from PIL import Image
from os.path import exists
from os import listdir, makedirs
import sys
sys.path.append("..")
sys.path.append("../src")
sys.path.append("../phantoms")
from src.algorithms_OhGreat.DART import *
from src.algorithms_OhGreat.SART import *
from src.algorithms_OhGreat.SIRT import *
from src.algorithms_OhGreat.FBP import *
from src.projections_OhGreat.project import *

def main():
    # total iterations for comparison algorithms
    iters = 10000
    # dart parameters
    dart_iters = 10
    rec_alg_iters = 1000
    p_fixed = 0.9
    # phantom family
    phants_fam = ["semilunars", "paws", "aliens", "clouds"]
    # define number of projections and angles
    n_projections = 50
    angle_range = 180
    noises = [ None, 1500, 3000, 4500, 6000, 7500, 9000]

    for phantoms in phants_fam:
        # input directory
        in_dir = f"../phantoms/{phantoms}/"
    
        for phantom_name in sorted(listdir(in_dir)):
            # skip already done experiments
            #if 'semilunar_0' in phantom_name or 'semilunar_1' in phantom_name:
            #    continue
            # output directory
            out_dir_noise = f"../results/noise_exps/{phantoms}/{phantom_name}/"
            if not exists(out_dir_noise):
                    makedirs(out_dir_noise)

            # choose a phantom
            phantom = np.array(Image.open(in_dir+phantom_name), dtype=np.uint8)
            img_width, img_height = phantom.shape
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"~Curr phantom: {phantom_name}~")

            noise_errors_sart = []
            noise_errors_sirt = []
            noise_errors_rbf = []
            noise_errors_dart_sart = []
            noise_errors_dart_fbp = []
            noise_errors_dart_sirt = []

            for noise in noises:
                print("~current noise level:", noise,"~")
                n_proj, n_detectors, det_spacing = n_projections, 512, 1
                vol_geom = astra.creators.create_vol_geom([img_width,img_height])
                phantom_id = astra.data2d.create('-vol', vol_geom, data=phantom)
                angles = np.linspace(0, angle_range, n_proj)
                projector_id, sino_id, sinogram = project_from_2D(phantom_id=phantom_id,
                                                                vol_geom=vol_geom,
                                                                n_projections=n_proj,
                                                                n_detectors=n_detectors,
                                                                detector_spacing=det_spacing,
                                                                angles=angles,
                                                                noise_factor=noise,
                                                                use_gpu=True)
                proj_geom = astra.create_proj_geom('parallel', det_spacing, 
                                                    n_detectors, angles)

                # SART
                _, sart_res = SART(vol_geom, 0, projector_id, sino_id, 
                                            iters, use_gpu=True)
                noise_errors_sart.append(np.abs(phantom - sart_res).mean())

                # SIRT
                _, sirt_res = SIRT(vol_geom, 0, sino_id, iters, use_gpu=True)
                noise_errors_sirt.append(np.abs(phantom - sirt_res).mean())

                # RBF
                _, fbp_res = FBP(vol_geom, 0, projector_id, sino_id, 
                                        iters, use_gpu=True)
                noise_errors_rbf.append(np.abs(phantom - fbp_res).mean())

                # DART with SART
                gray_lvls = np.unique(phantom).astype(np.float32)
                d = DART(gray_levels=gray_lvls, p=p_fixed, rec_shape=phantom.shape,
                    proj_geom=proj_geom, projector_id=projector_id,
                    sinogram=sinogram)
                # run the algorithm
                dart_res = d.run(iters=dart_iters,rec_alg="SART_CUDA",rec_iter=rec_alg_iters)
                noise_errors_dart_sart.append(np.abs(phantom - dart_res).mean())

                # DART with SIRT
                gray_lvls = np.unique(phantom).astype(np.float32)
                d = DART(gray_levels=gray_lvls, p=p_fixed, rec_shape=phantom.shape,
                    proj_geom=proj_geom, projector_id=projector_id,
                    sinogram=sinogram)
                # run the algorithm
                dart_res = d.run(iters=dart_iters,rec_alg="SIRT_CUDA",rec_iter=rec_alg_iters)
                noise_errors_dart_sirt.append(np.abs(phantom - dart_res).mean())

                # DART with FBP
                d = DART(gray_levels=gray_lvls, p=p_fixed, rec_shape=phantom.shape,
                    proj_geom=proj_geom, projector_id=projector_id,
                    sinogram=sinogram)
                # run the algorithm
                dart_res = d.run(iters=dart_iters,rec_alg="FBP_CUDA",rec_iter=rec_alg_iters)
                noise_errors_dart_fbp.append(np.abs(phantom - dart_res).mean())

            np.save(out_dir_noise+"SART", noise_errors_sart)
            np.save(out_dir_noise+"SIRT", noise_errors_sirt)
            np.save(out_dir_noise+"RBF", noise_errors_rbf)
            np.save(out_dir_noise+"DART_sart", noise_errors_dart_sart)
            np.save(out_dir_noise+"DART_fbp", noise_errors_dart_fbp)
            np.save(out_dir_noise+"DART_sirt", noise_errors_dart_sirt)

            # free memory
            astra.data2d.clear()
            astra.projector.clear()
            astra.algorithm.clear()            

if __name__ == "__main__":
    main()