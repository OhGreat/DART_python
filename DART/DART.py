import numpy as np

class DART:
    """ gray_levels: R = {ρ_1, ..,  ρ_l}

    """
    def __init__(self, gray_levels):
        self.gray_levels = gray_levels
        self.thresholds = [(gray_levels[i]+gray_levels[i+1])/2 
                            for i in range(len(gray_levels)-1) ]

    def threshold(self, img):
        for thresh_idx in range(len(self.thresholds)):
            if thresh_idx == 0:
                img[img < self.thresholds[thresh_idx]] = self.gray_levels[thresh_idx]
            elif thresh_idx >= len(self.thresholds)-1:
                img[img > self.threshold[thresh_idx]] = self.gray_levels[thresh_idx]
            else:
                img[(img < self.thresholds[thresh_idx]) and 
                    (img > self.thresholds[thresh_idx-1])] = self.gray_levels[[thresh_idx]]

        return img


dart = DART()

gray_lvls = [0, 125, 255]

#img = np.random