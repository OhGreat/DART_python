import numpy as np
from PIL import Image
from src.algorithms import DART
from src.projections import project_from_2D
from os import listdir, makedirs
from os.path import exists

def main():
    # output directory
    out_dir = "results/semilunars"
    if not exists(out_dir):
        makedirs(out_dir)

    dir = "phantoms/semilunars/"
    for filename in sorted(listdir(dir)):
        phantom = np.array(Image.open(dir+filename), dtype=np.uint8)
        stats = (5, phantom ,out_dir)
        print(stats)




if __name__ == "__main__":
    main()