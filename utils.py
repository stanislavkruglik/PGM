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

def alpha_beta_swap(psi1, psi2V, psi2H, metric, max_iter=500, display=False):
    k, n, m = psi1.shape
    # Init
    states = np.random.randint(0, k, size=(n, m))
    alpha = 0
    beta = 1
    best_energy = np.inf
    xx, yy = np.indices((n,m))

    for iter in range(max_iter):
        print(iter)
        alpha = (iter/(k-1)) % k
        beta = (k-1  - iter) % k
 
        mask = (states == alpha) + (states == beta)

        up = np.zeros((n, m))
        up[mask] = psi1[alpha, mask]

        down = np.zeros((n, m))
        down[mask] = psi1[beta, mask]

        up[:,1:] += psi2H[states[:,:-1], alpha, xx[:,:-1], yy[:,:-1]] * metric[states[:,:-1], alpha] * mask[:,1:] * (~mask)[:,:-1]
        up[:,:-1] += psi2H[alpha, states[:,1:], xx[:,:-1], yy[:,:-1]] * metric[alpha, states[:,1:]]* mask[:,:-1] * (~mask)[:,1:]
        up[1:] += psi2V[states[:-1], alpha, xx[:-1], yy[:-1]] * metric[states[:-1], alpha] * mask[1:] * (~mask)[:-1]
        up[:-1] += psi2V[alpha, states[1:], xx[:-1], yy[:-1]] * metric[alpha, states[1:]] * mask[:-1] * (~mask)[1:]

        down[:,1:] += psi2H[states[:,:-1], beta, xx[:,:-1], yy[:,:-1]] * metric[states[:,:-1], beta] * mask[:,1:] * (~mask)[:,:-1]
        down[:,:-1] += psi2H[beta, states[:,1:], xx[:,:-1], yy[:,:-1]] * metric[beta, states[:,1:]] * mask[:,:-1] * (~mask)[:,1:]
        down[1:] += psi2V[states[:-1], beta, xx[:-1], yy[:-1]] * metric[states[:-1], beta] * mask[1:] * (~mask)[:-1]
        down[:-1] += psi2V[beta, states[1:], xx[:-1], yy[:-1]] * metric[beta, states[1:]] * mask[:-1] * (~mask)[1:]

        hor01 = np.zeros((n, m-1))
        hor10 = np.zeros((n, m-1))
        hor01[mask[:,1:] * mask[:,:-1]] = psi2H[alpha, beta, mask[:,1:] * mask[:,:-1]].ravel() * metric[alpha, beta]
        hor10[mask[:,1:] * mask[:,:-1]] = psi2H[beta, alpha, mask[:,1:] * mask[:,:-1]].ravel() * metric[beta, alpha]

        ver01 = np.zeros((n-1, m))
        ver10 = np.zeros((n-1, m))
        ver01[mask[1:] * mask[:-1]] = psi2V[alpha, beta, mask[1:] * mask[:-1]].ravel() * metric[alpha, beta]
        ver10[mask[1:] * mask[:-1]] = psi2V[beta, alpha, mask[1:] * mask[:-1]].ravel() * metric[beta, alpha]

        deltaH = (hor01 - hor10) / 2.
        up[:, :-1] += deltaH
        up[:, 1:] += deltaH

        deltaV = (ver01 - ver10) / 2.
        up[:-1] += deltaV
        up[1:] += deltaV
        hor01 = (hor01 + hor10) / 2.
        ver01 = (ver01 + ver10) / 2.
        terminal_weights = np.vstack((up.ravel(), down.ravel())).T

        indexation = np.arange(n * m).reshape(n, m)
        hor_weights = np.vstack((indexation[:, :-1].ravel(), indexation[:, 1:].ravel(), hor01.ravel(), hor01.ravel()))
        ver_weights = np.vstack((indexation[:-1].ravel(), indexation[1:].ravel(), ver01.ravel(), ver01.ravel()))
        edge_weights = np.hstack((hor_weights, ver_weights)).T

        cut, curr_labels = graph_cut.graph_cut(terminal_weights, edge_weights)

        curr_labels = curr_labels.reshape((n, m))
        states[(curr_labels==0) * mask] = beta
        states[(curr_labels==1) * mask] = alpha
        E = psi1[states, xx, yy].sum() + (psi2H * metric[states[:,:-1], states[:,1:]]).sum() + (psi2V * metric[states[:-1], states[1:]]).sum()
        if iter:
            if (labels_old != states).sum() < n*m/200:
                break
        labels_old = states.copy()

    return states.copy().ravel()

def collage(images, masks, method='a', display=True, potential= lambda x: x.std(0).mean(2).ravel()):
    eps = 1e-5
    images = np.array(np.atleast_3d(images))
    k, n, m = images.shape[:3]
    # Metric
    metric = np.ones((k,k)) - np.eye(k)
    if method == 'a':
        # psi1 potential
        psi1 = np.inf*np.ones((n, m, k))
        for i, mask in enumerate(masks):
            psi1[mask, i] = 0
        # Pairwise potentials
        dV = (images[:, 1:] - images[:, :-1]) + eps
        dH = (images[:, :, 1:] - images[:, :, :-1]) + eps
        psi2V = potential(dV)
        psi2H = potential(dH)    
        mask = alpha_expantion_routine(psi1, psi2V, psi2H, metric, max_iter=2 * k, display=display)
    else:
        # psi1 potential
        psi1 = np.zeros((k, n, m))
        for i, mask in enumerate(masks):
            psi1[:, mask] = 1e10
            psi1[i, mask] = 0
        # Pairwise potentials
        psi2V = np.empty((k, k, n-1, m), dtype=float)
        psi2H = np.empty((k, k, n, m-1), dtype=float)
        for im in range(k):
            psi2V[im] = np.exp(-((np.fabs(images[im, :-1] - images[:, :-1]).sum(axis=3) +
                             np.fabs(images[im, 1:] - images[:, 1:]).sum(axis=3)))) + eps
            psi2H[im] = np.exp(-((np.fabs(images[im, :, :-1] - images[:, :, :-1]).sum(axis=3) +
                             np.fabs(images[im, :, 1:] - images[:, :, 1:]).sum(axis=3)))) + eps
        mask = alpha_beta_swap(psi1, psi2V, psi2H, metric, max_iter = 100, display=display)    
    image = images.reshape(k, -1, 3)[mask, np.arange(n * m)]
    image = image.reshape(n, m, 3)
    return image, mask