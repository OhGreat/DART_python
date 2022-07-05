import astra
import numpy as np
from PIL import Image
from os.path import exists
from os import listdir, makedirs
import sys
sys.path.append("..")
sys.path.append("../src")
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
    n_projections = [2, 4, 6, 8, 10, 12, 14, 16, 20]
    angle_range = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180]
    
    for phantoms in phants_fam:
        # input directory
        in_dir = f"../phantoms/{phantoms}/"
    
        for phantom_name in sorted(listdir(in_dir)):
            # output directory
            out_dir_proj = f"../results/n_proj/{phantoms}/{phantom_name}/"
            out_dir_angles = f"../results/angle_range/{phantoms}/{phantom_name}/"
            if not exists(out_dir_proj):
                    makedirs(out_dir_proj)
            if not exists(out_dir_angles):
                    makedirs(out_dir_angles)

            # choose a phantom
            phantom = np.array(Image.open(in_dir+phantom_name), dtype=np.uint8)
            img_width, img_height = phantom.shape
            print(f"Curr phantom: {phantom_name}")

            proj_errors_sart = []
            proj_errors_sirt = []
            proj_errors_rbf = []
            proj_errors_dart_sart = []
            proj_errors_dart_fbp = []
            proj_errors_dart_sirt = []

            for curr_proj in n_projections:
                print("curr_proj:", curr_proj)
                n_proj, n_detectors, det_spacing = curr_proj, 512, 1
                vol_geom = astra.creators.create_vol_geom([img_width,img_height])
                phantom_id = astra.data2d.create('-vol', vol_geom, data=phantom)
                angles = np.linspace(0, np.pi, curr_proj)
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
                proj_errors_sart.append(np.abs(phantom - sart_res).mean())

                # SIRT
                _, sirt_res = SIRT(vol_geom, 0, sino_id, iters, use_gpu=True)
                proj_errors_sirt.append(np.abs(phantom - sirt_res).mean())

                # RBF
                _, fbp_res = FBP(vol_geom, 0, projector_id, sino_id, 
                                        iters, use_gpu=True)
                proj_errors_rbf.append(np.abs(phantom - fbp_res).mean())

                # DART with SART
                gray_lvls = np.unique(phantom).astype(np.float32)
                d = DART(gray_levels=gray_lvls, p=p_fixed, rec_shape=phantom.shape,
                    proj_geom=proj_geom, projector_id=projector_id,
                    sinogram=sinogram)
                # run the algorithm
                dart_res = d.run(iters=dart_iters,rec_alg="SART_CUDA",rec_iter=rec_alg_iters)
                proj_errors_dart_sart.append(np.abs(phantom - dart_res).mean())

                # DART with SIRT
                gray_lvls = np.unique(phantom).astype(np.float32)
                d = DART(gray_levels=gray_lvls, p=p_fixed, rec_shape=phantom.shape,
                    proj_geom=proj_geom, projector_id=projector_id,
                    sinogram=sinogram)
                # run the algorithm
                dart_res = d.run(iters=dart_iters,rec_alg="SIRT_CUDA",rec_iter=rec_alg_iters)
                proj_errors_dart_sirt.append(np.abs(phantom - dart_res).mean())

                # DART with FBP
                d = DART(gray_levels=gray_lvls, p=p_fixed, rec_shape=phantom.shape,
                    proj_geom=proj_geom, projector_id=projector_id,
                    sinogram=sinogram)
                # run the algorithm
                dart_res = d.run(iters=dart_iters,rec_alg="FBP_CUDA",rec_iter=rec_alg_iters)
                proj_errors_dart_fbp.append(np.abs(phantom - dart_res).mean())

            np.save(out_dir_proj+"SART", proj_errors_sart)
            np.save(out_dir_proj+"SIRT", proj_errors_sirt)
            np.save(out_dir_proj+"RBF", proj_errors_rbf)
            np.save(out_dir_proj+"DART_sart", proj_errors_dart_sart)
            np.save(out_dir_proj+"DART_fbp", proj_errors_dart_fbp)
            np.save(out_dir_proj+"DART_sirt", proj_errors_dart_sirt)
            
            # Number of angles experiments
            ang_errors_sart = []
            ang_errors_sirt = []
            ang_errors_rbf = []
            ang_errors_dart_sart = []
            ang_errors_dart_fbp = []
            ang_errors_dart_sirt = []

            for curr_ang in angle_range:
                print("curr angle:", curr_ang)
                n_proj, n_detectors, det_spacing = 14, 512, 1
                vol_geom = astra.creators.create_vol_geom([img_width,img_height])
                phantom_id = astra.data2d.create('-vol', vol_geom, data=phantom)
                angles = np.linspace(0, np.pi*(curr_ang/180), n_proj)
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
                ang_errors_sart.append(np.abs(phantom - sart_res).mean())

                # SIRT
                _, sirt_res = SIRT(vol_geom, 0, sino_id, iters, use_gpu=True)
                ang_errors_sirt.append(np.abs(phantom - sirt_res).mean())

                # FBP
                _, fbp_res = FBP(vol_geom, 0, projector_id, sino_id, 
                                        iters, use_gpu=True)
                ang_errors_rbf.append(np.abs(phantom - fbp_res).mean())

                # instanciate DART with SIRT
                gray_lvls = np.unique(phantom).astype(np.float32)
                d = DART(gray_levels=gray_lvls, p=p_fixed, rec_shape=phantom.shape,
                    proj_geom=proj_geom, projector_id=projector_id,
                    sinogram=sinogram)
                # run the algorithm
                dart_res = d.run(iters=dart_iters,rec_alg="SIRT_CUDA",rec_iter=rec_alg_iters)
                ang_errors_dart_sirt.append(np.abs(phantom - dart_res).mean())

                # instanciate DART with SART
                gray_lvls = np.unique(phantom).astype(np.float32)
                d = DART(gray_levels=gray_lvls, p=p_fixed, rec_shape=phantom.shape,
                    proj_geom=proj_geom, projector_id=projector_id,
                    sinogram=sinogram)
                # run the algorithm
                dart_res = d.run(iters=dart_iters,rec_alg="SART_CUDA",rec_iter=rec_alg_iters)
                ang_errors_dart_sart.append(np.abs(phantom - dart_res).mean())

                # instanciate DART with FBP
                d = DART(gray_levels=gray_lvls, p=p_fixed, rec_shape=phantom.shape,
                    proj_geom=proj_geom, projector_id=projector_id,
                    sinogram=sinogram)
                # run the algorithm
                dart_res = d.run(iters=dart_iters,rec_alg="FBP_CUDA",rec_iter=rec_alg_iters)
                ang_errors_dart_fbp.append(np.abs(phantom - dart_res).mean())

                astra.data2d.clear()
                astra.projector.clear()
                astra.algorithm.clear()

            np.save(out_dir_angles+f"SART_{phantoms}", ang_errors_sart)
            np.save(out_dir_angles+f"SIRT_{phantoms}", ang_errors_sirt)
            np.save(out_dir_angles+f"RBF_{phantoms}", ang_errors_rbf)
            np.save(out_dir_angles+f"DART_sart_{phantoms}", ang_errors_dart_sart)
            np.save(out_dir_angles+f"DART_sirt_{phantoms}", ang_errors_dart_sirt)
            np.save(out_dir_angles+f"DART_fbp_{phantoms}", ang_errors_dart_fbp)

if __name__ == "__main__":
    main()