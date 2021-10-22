import numpy as np
import os
import cohere.utilities.utils as ut

def run_AI(data, threshold, sigma, dir):
    print('AI guess')

    guess = np.random.rand(*data.shape)
    np.save(os.path.join(dir, 'image.npy'), guess)

    support = ut.shrink_wrap(guess, threshold, sigma)
    np.save(os.path.join(dir, 'support.npy'), support)