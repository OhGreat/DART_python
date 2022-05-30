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
    # phantom family
    phantom_fam = ["semilunars", "aliens", "paws"]
    for phantom_family in phantom_fam:
        # input directory
        in_dir = f"phantoms/{phantom_family}/"
        # output directory
        results_dir = f"results/phantoms/{phantom_family}/"
        if not exists(results_dir):
            makedirs(results_dir)

        all_errors = []
        curr_file = 0
        for filename in sorted(listdir(in_dir)):
            # current phantom reconstruction errors
            curr_errors = []
            # open phantom
            phantom = np.array(Image.open(in_dir+filename), dtype=np.uint8)
            img_width, img_height = phantom.shape

            n_proj, n_detectors, det_spacing, ang_mul = 50, 512, 1, 1
            vol_geom = astra.creators.create_vol_geom([img_width,img_height])
            phantom_id = astra.data2d.create('-vol', vol_geom, data=phantom)
            projector_id, sino_id, _ = project_from_2D(phantom_id, 
                                                            vol_geom,n_proj,
                                                            n_detectors,
                                                            det_spacing,
                                                            ang_mul,
                                                            use_gpu=True)
            # SART
            _, sart_res = DART().SART(vol_geom, 0, projector_id, sino_id, 
                                        iters, use_gpu=True)
            sart_res[sart_res > 255] = 255
            sart_res[sart_res < 0] = 0
            curr_errors.append(np.abs(phantom - sart_res).mean())
            Image.fromarray(sart_res.astype(np.uint8)).save(results_dir+f"SART_{curr_file}.png")

            # SIRT
            _, sirt_res = DART().SIRT(vol_geom, 0, sino_id, 
                                    iters, use_gpu=True)
            sirt_res[sirt_res > 255] = 255
            sirt_res[sirt_res < 0] = 0
            curr_errors.append(np.abs(phantom - sirt_res).mean())
            Image.fromarray(sirt_res.astype(np.uint8)).save(results_dir+f"SIRT_{curr_file}.png")

            # FBP
            _, fbp_res = DART().FBP(vol_geom, 0, projector_id, sino_id, 
                                        iters, use_gpu=True)
            fbp_res[fbp_res > 255] = 255
            fbp_res[fbp_res < 0] = 0
            curr_errors.append(np.abs(phantom - fbp_res).mean())
            Image.fromarray(fbp_res.astype(np.uint8)).save(results_dir+f"FBP_{curr_file}.png")


            # DART 0
            dart = DART()
            dart_res = dart(iters=100,
                    gray_levels=np.unique(phantom).astype(np.float32),
                    p=0.9, vol_shape=phantom.shape,
                    projector_id=projector_id, sino_id=sino_id,
                    rec_algs=("SART", "SART"),
                    rec_iter=200, use_gpu=True)
            dart_res[dart_res > 255] = 255
            dart_res[dart_res < 0] = 0
            curr_errors.append(np.abs(phantom - dart_res).mean())
            Image.fromarray(dart_res.astype(np.uint8)).save(results_dir+f"DART_SART_SART_{curr_file}.png")

            # DART 1
            dart = DART()
            dart_res = dart(iters=100,
                    gray_levels=np.unique(phantom).astype(np.float32),
                    p=0.9, vol_shape=phantom.shape,
                    projector_id=projector_id, sino_id=sino_id,
                    rec_algs=("FBP", "FBP"),
                    rec_iter=200, use_gpu=True)
            dart_res[dart_res > 255] = 255
            dart_res[dart_res < 0] = 0
            curr_errors.append(np.abs(phantom - dart_res).mean())
            Image.fromarray(dart_res.astype(np.uint8)).save(results_dir+f"DART_FBP_FBP_{curr_file}.png")

            # DART 2
            dart = DART()
            dart_res = dart(iters=100,
                    gray_levels=np.unique(phantom).astype(np.float32),
                    p=0.9, vol_shape=phantom.shape,
                    projector_id=projector_id, sino_id=sino_id,
                    rec_algs=("SIRT", "FBP"),
                    rec_iter=200, use_gpu=True)
            dart_res[dart_res > 255] = 255
            dart_res[dart_res < 0] = 0
            curr_errors.append(np.abs(phantom - dart_res).mean())
            Image.fromarray(dart_res.astype(np.uint8)).save(results_dir+f"DART_SIRT_FBP_{curr_file}.png")

            # DART 3
            dart = DART()
            dart_res = dart(iters=100,
                    gray_levels=np.unique(phantom).astype(np.float32),
                    p=0.9, vol_shape=phantom.shape,
                    projector_id=projector_id, sino_id=sino_id,
                    rec_algs=("SART", "FBP"),
                    rec_iter=200, use_gpu=True)
            dart_res[dart_res > 255] = 255
            dart_res[dart_res < 0] = 0
            curr_errors.append(np.abs(phantom - dart_res).mean())
            Image.fromarray(dart_res.astype(np.uint8)).save(results_dir+f"DART_SART_FBP_{curr_file}.png")

            curr_file += 1
            all_errors.append(curr_errors)
            
            astra.data2d.clear()
            astra.projector.clear()
            astra.algorithm.clear()

        # save errors as npy
        np.save(results_dir+f"err_sart_sirt_fbp_dart0_dart1_dart2_dart3" ,all_errors)
        curr_file = 0
        print(all_errors)
        all_errors = []

if __name__ == "__main__":
    main()