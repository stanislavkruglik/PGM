import numpy as np
from graph_cut import graph_cut
import matplotlib.pyplot as plt

def alpha_expantion_routine(psi1, psi2V, psi2H, metric, max_iter=500, display=False):
    n, m, k = psi1.shape
    psi1 = psi1.reshape(n * m, k)
    # Init
    states = np.random.randint(0, k, size=(n, m))
   
    current_state = 0 # Alpha
    for iter_ in range(max_iter):
        current_state = (current_state + 1) % k

        up = psi1[np.arange(n * m), states.ravel()].reshape(n, m)
        up[states == current_state] = np.inf
        down = psi1[np.arange(n * m), current_state].reshape(n, m)
        # Horizontal 
        hor00 = psi2H * metric[states[:, :-1].ravel(), states[:, 1:].ravel()]
        hor01 = psi2H * metric[states[:, :-1].ravel(), current_state]
        hor10 = psi2H * metric[current_state, states[:, 1:].ravel()]
        
        up_h = (hor01 - hor10 + hor00).reshape((n, m - 1))
        up[:, :-1] +=  up_h / 2
        up[:, 1:] += up_h / 2
        # Vertical 
        vert00 = psi2V * metric[states[:-1, :].ravel(), states[1:, :].ravel()]
        vert01 = psi2V * metric[states[:-1, :].ravel(), current_state]
        vert10 = psi2V * metric[current_state, states[1:, :].ravel()]

        up_v = (vert01 - vert10 + vert00).reshape((n - 1, m))
        up[:-1, :] += up_v / 2
        up[1:, :] += up_v / 2
        # Cut
        v1 = np.arange((n - 1) * m)
        v2 = m + np.arange((n - 1) * m)
        h1 = np.arange(n * m).reshape((n, m))[:, :-1].ravel()
        h2 = np.arange(n * m).reshape((n, m))[:, 1:].ravel()
        hor01 = (hor01 + hor10 - hor00) / 2
        vert01 = (vert01 + vert10 - vert00) / 2

        vert_weights = np.vstack([v1, v2, vert01, vert01])
        hor_weights = np.vstack([h1, h2, hor01, hor01])  

        edge_weights = np.hstack((vert_weights, hor_weights)).T
        terminal_weights = np.vstack((up.ravel(), down.ravel())).T     
        cut, states_ = graph_cut(terminal_weights, edge_weights)

        states_ = states_.reshape((n, m))
        states[states_ == 0] = current_state

        # Display routine
        if display:   
        	plt.figure(figsize=((6,6)))
        	plt.title('Iter:'+str(iter_))
	        plt.imshow(states, cmap="gray")
	        plt.yticks([])
    		plt.xticks([])
	        plt.show()
    states = states.ravel()
    return states


def collage(images, masks, method='a', display=True):
    eps = 1e-5
    images = np.array(np.atleast_3d(images))
    k, n, m = images.shape[:3]
    # Metric
    metric = np.ones((k,k)) - np.eye(k)
    # psi1 potential
    psi1 = np.inf*np.ones((n, m, k))
    for i, mask in enumerate(masks):
        psi1[mask, i] = 0
    # Pairwise potentials
    dV = (images[:, 1:] - images[:, :-1]) + eps
    dH = (images[:, :, 1:] - images[:, :, :-1]) + eps
    psi2V = dV.std(0).mean(2).ravel()
    psi2H = dH.std(0).mean(2).ravel()

    if method == 'a':
        mask = alpha_expantion_routine(psi1, psi2V, psi2H, metric, max_iter=2 * k, display=display)
    else:
        print "ab method is not implemented"
        
    image = images.reshape(k, -1, 3)[mask, np.arange(n * m)]
    image = image.reshape(n, m, 3)
    return image, mask