import argparse
import astra
import random
import numpy as np
from PIL import Image
from os.path import exists
from os import listdir, makedirs
import sys
import argparse
sys.path.append("..")
sys.path.append("../src")
sys.path.append("../phantoms")
from src.algs import *
from src.projections import *

def main():
    parser = argparse.ArgumentParser()
    # 0: mixed approach, 1 overestimate, 2: underestimate
    parser.add_argument('-type', action='store',
                        dest='exp_name', type=int,
                        default=0,
                        help="Defines the type of the experiment.")
    parser.add_argument('-exp_name', action='store',
                        dest='exp_name', type=str,
                        default='test_experiment',
                        help="Defines the name of the experiment.")
    args = parser.parse_args()

    # total iterations for comparison algorithms
    iters = 10000
    # dart parameters
    dart_iters = 10
    rec_alg_iters = 1000
    p_fixed = 0.9
    # phantom family
    phantom_families = ["semilunars", "paws", "aliens", "clouds"]
    # define number of projections and angles
    n_projections = 50
    angle_range = 180
    gray_val_noise = [0.02, 0.06, 0.1, 0.14, 0.18, 0.22, 0.26, 0.3, 0.34, 0.38, 0.4]
    base_in_dir = "../phantoms/"
    base_out_dir = f"../results/{args.exp_name}/"

    for curr_phantom in phantom_families:
        # input directory
        in_dir = base_in_dir + f"{curr_phantom}/"
    
        for phantom_name in sorted(listdir(in_dir)):
            # skip already done experiments
            #if 'semilunar_0' in phantom_name or 'semilunar_1' in phantom_name:
            #    continue
            # output directory
            out_dir_noise = base_out_dir + f"{curr_phantom}/{phantom_name}/"
            if not exists(out_dir_noise):
                    makedirs(out_dir_noise)

            # choose a phantom
            phantom = np.array(Image.open(in_dir+phantom_name), dtype=np.uint8)
            img_width, img_height = phantom.shape
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"~ curr phantom: {phantom_name} ~")

            gray_val_err_sart = []
            gray_val_err_sirt = []
            gray_val_err_rbf = []
            gray_val_err_dart_sart = []
            gray_val_err_dart_fbp = []
            gray_val_err_dart_sirt = []

            for noise_mul in gray_val_noise:
                # create noised gray values
                phant_grays = np.unique(phantom).astype(np.float32)
                phant_grays[phant_grays == 0] = 1
                # mixed overestimation and underestimation
                if args.type == 0:
                    signs = np.array([1 if random.random() > 0.5 else -1 for _ in range(len(phant_grays))])
                    noised_gray = phant_grays + (1 + phant_grays)*signs*noise_mul
                    noised_gray =noised_gray.clip(0,255.)
                elif args.type == 1:
                    # only overestimate
                    noised_gray = phant_grays + (1 + phant_grays)*noise_mul
                elif args.type == 2:
                    # only underestimate
                    noised_gray = phant_grays - (1 + phant_grays)*noise_mul
                else:
                    exit
                print(f"~ current gray_values: {noised_gray}, original: {phant_grays} ~")
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
                                                                noise_factor=None,
                                                                use_gpu=True)
                proj_geom = astra.create_proj_geom('parallel', det_spacing, 
                                                    n_detectors, angles)

                # SART
                _, sart_res = SART(vol_geom, 0, projector_id, sino_id, 
                                            iters, use_gpu=True)
                gray_val_err_sart.append(np.abs(phantom - sart_res).mean())

                # SIRT
                _, sirt_res = SIRT(vol_geom, 0, sino_id, iters, use_gpu=True)
                gray_val_err_sirt.append(np.abs(phantom - sirt_res).mean())

                # RBF
                _, fbp_res = FBP(vol_geom, 0, projector_id, sino_id, 
                                        iters, use_gpu=True)
                gray_val_err_rbf.append(np.abs(phantom - fbp_res).mean())

                # DART with SART
                d = DART(gray_levels=noised_gray, p=p_fixed, rec_shape=phantom.shape,
                    proj_geom=proj_geom, projector_id=projector_id,
                    sinogram=sinogram)
                # run the algorithm
                dart_res = d.run(iters=dart_iters,rec_alg="SART_CUDA",rec_iter=rec_alg_iters)
                gray_val_err_dart_sart.append(np.abs(phantom - dart_res).mean())

                # DART with SIRT
                d = DART(gray_levels=noised_gray, p=p_fixed, rec_shape=phantom.shape,
                    proj_geom=proj_geom, projector_id=projector_id,
                    sinogram=sinogram)
                # run the algorithm
                dart_res = d.run(iters=dart_iters,rec_alg="SIRT_CUDA",rec_iter=rec_alg_iters)
                gray_val_err_dart_sirt.append(np.abs(phantom - dart_res).mean())

                # DART with FBP
                d = DART(gray_levels=noised_gray, p=p_fixed, rec_shape=phantom.shape,
                    proj_geom=proj_geom, projector_id=projector_id,
                    sinogram=sinogram)
                # run the algorithm
                dart_res = d.run(iters=dart_iters,rec_alg="FBP_CUDA",rec_iter=rec_alg_iters)
                gray_val_err_dart_fbp.append(np.abs(phantom - dart_res).mean())

            np.save(out_dir_noise+"SART", gray_val_err_sart)
            np.save(out_dir_noise+"SIRT", gray_val_err_sirt)
            np.save(out_dir_noise+"RBF", gray_val_err_rbf)
            np.save(out_dir_noise+"DART_sart", gray_val_err_dart_sart)
            np.save(out_dir_noise+"DART_fbp", gray_val_err_dart_fbp)
            np.save(out_dir_noise+"DART_sirt", gray_val_err_dart_sirt)

            # free memory
            astra.data2d.clear()
            astra.projector.clear()
            astra.algorithm.clear()            

if __name__ == "__main__":
    main()