import astra

def FBP(vol_geom, vol_data, projector_id, sino_id, iters=2000, use_gpu=False):
    # create starting reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom, data=vol_data)
    # define SART configuration parameters
    alg_cfg = astra.astra_dict('FBP_CUDA' if use_gpu else 'FBP')
    alg_cfg['ProjectorId'] = projector_id
    alg_cfg['ProjectionDataId'] = sino_id
    alg_cfg['ReconstructionDataId'] = rec_id
    alg_cfg['option'] = {}
    alg_cfg['option']['MinConstraint'] = 0
    alg_cfg['option']['MaxConstraint'] = 255
    # define algorithm
    algorithm_id = astra.algorithm.create(alg_cfg)
    # run the algirithm
    astra.algorithm.run(algorithm_id, iters)
    # create reconstruction data
    rec = astra.data2d.get(rec_id)

    return rec_id, rec