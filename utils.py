import numpy as np
import graph_cut


def collage(images, masks, method='a'):
    eps = 1e-5
    images = np.array(np.atleast_3d(images))
    k, n, m = images.shape[:3]
       
    # Metric
    metric = np.ones((k,k)) - np.eye(k)
    
    # Unary potential
    psi1 = np.zeros((n, m, k))
    for i, mask in enumerate(masks):
        psi1[seed, :] = np.inf
        psi1[seed, i] = 0
     
    # Pairwise potentials
    dV = (images[:, 1:] - images[:, :-1]) + eps
    dH = (images[:, :, 1:] - images[:, :, :-1]) + eps
    psi2V = dV.std(0).mean(2)
    psi2H = dH.std(0).mean(2)
 
    if method == 'a':
        mask, energy = alpha_expantion_routine(psi1, psi2V, psi2H, metric, maxIter=2 * k)
    else:
        print "ab method is not implemented"
        
    image = images.reshape(k, -1, 3)[mask, np.arange(n * m)]
    image = image.reshape(n, m, 3)

    return image, mask