
# coding: utf-8

# In[20]:
import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import graph_cut
import random
from skimage import io, color
        
# get_ipython().magic(u'matplotlib inline')


# In[21]:

def get_masks(masks_urls):
    masks = []
    for url in masks_urls:
        image = io.imread(url, 1)
        mask = np.zeros(image.shape, dtype=bool)
        image = np.array(image)
        mask[image < 0.5] = True
        masks.append(mask)
    return np.array(masks)


# In[22]:

def get_images(urls):
    images = []
    for url in urls:
        B = io.imread(url)
        B = np.array(B)
        images.append(B)
    return np.array(images)


# In[23]:

def get_metric(number_images):
    return 1 - np.eye(number_images, dtype=int)

def get_unary(images, masks):
    unary = np.zeros(images[0].shape[:2] + (len(images),))
    unary[:,:, :] = 1000
    for mask_id in range(len(masks)):
        unary[:,:, mask_id][masks[mask_id]] = 0
    return unary


# In[24]:

def get_C(images, masks):
    k, n, m, _ = images.shape
    vertC = np.zeros((n - 1, m))
    horC = np.zeros((n, m - 1))
    for i in range(n):
        for j in range(m):
            if i < n - 1:
                pix1 = images[:, i, j]
                pix2 = images[:, i + 1, j]
                vertC[i][j] = np.exp(-np.linalg.norm(pix1 - pix2, axis=1).mean())
            if j < m - 1:
                pix1 = images[:, i, j]
                pix2 = images[:, i, j + 1]
                horC[i][j] = np.exp(-np.linalg.norm(pix1 - pix2, axis=1).mean())
    return vertC, horC


# In[25]:

def vertex_id(i, j, n, m):
    return i*m + j

def teta(alpha, unary, mask, (i, j), value):
    if value == 1:
        return unary[i, j, alpha]
    if mask[i][j] == alpha:
        return 10000000
    else:
        return unary[i, j, mask[i][j]]
    
def doAlphaExpansion(alpha, mask, n, m, unary, vertC, horC, metric):
    
    terminal_weights = []
    edge_weights = []
    
    teta_unary = np.zeros((n, m, 2))
    pair_test_hor = np.zeros((n, m, 2, 2))
    pair_test_vert = np.zeros((n, m, 2, 2))
    for i in range(n):
        for j in range(m):
            teta_unary[i, j, 0] += teta(alpha, unary, mask, (i, j), 0)
            teta_unary[i, j, 1] += teta(alpha, unary, mask, (i, j), 1)
            
            #Ver
            if i < n - 1:
                # (i, j), (i + 1, j)
                teta_pair = np.zeros((2, 2))                
                teta_pair[0, 0] = vertC[i, j]*metric[mask[i,j], mask[i+1,j]]
                teta_pair[1, 0] = vertC[i, j]*metric[alpha, mask[i+1,j]]
                teta_pair[0, 1] = vertC[i, j]*metric[mask[i,j], alpha]
                teta_pair[1, 1] = vertC[i, j]*metric[alpha, alpha]
                               
                c_i_j = (teta_pair[0, 1]+teta_pair[1, 0]-teta_pair[0, 0]-teta_pair[1, 1]) / 2.
                delta = (teta_pair[0, 1]-teta_pair[0, 0])-(teta_pair[1, 0] - teta_pair[1, 1])
                teta_unary[i, j, 0] += teta_pair[0, 0]
                teta_unary[i, j, 1] += teta_pair[1, 1]
                teta_unary[i+1, j, 1] += delta / 2.
                teta_unary[i, j, 1] -= delta / 2.
                
                edge_weights.append([
                    vertex_id(i, j, n, m), 
                    vertex_id(i+1, j, n, m),
                    c_i_j,
                    c_i_j])
            #Hor
            if j < m - 1:
                # (i, j), (i, j + 1)
                teta_pair = np.zeros((2, 2))                
                teta_pair[0, 0] = horC[i, j]*metric[mask[i,j], mask[i,j+1]]
                teta_pair[1, 0] = horC[i, j]*metric[alpha, mask[i,j+1]]
                teta_pair[0, 1] = horC[i, j]*metric[mask[i,j], alpha]
                teta_pair[1, 1] = horC[i, j]*metric[alpha, alpha]
                
                c_i_j = (teta_pair[0, 1]+teta_pair[1, 0]-teta_pair[0, 0]-teta_pair[1, 1]) / 2.
                delta = (teta_pair[0, 1]-teta_pair[0, 0])-(teta_pair[1, 0] - teta_pair[1, 1])
                teta_unary[i, j, 0] += teta_pair[0, 0]
                teta_unary[i, j, 1] += teta_pair[1, 1]
                teta_unary[i, j+1, 1] += delta / 2.
                teta_unary[i, j, 1] -= delta / 2.
                
                edge_weights.append([
                    vertex_id(i, j, n, m), 
                    vertex_id(i, j+1, n, m),
                    c_i_j,
                    c_i_j
                ])
                
    delta_energy = 0
    
    for i in range(n):
        for j in range(m):
            if np.min(teta_unary[i, j, :]) < 0:
                d = np.min(teta_unary[i, j, :])
                teta_unary[i, j, :] -= d
                delta_energy += d
    
    for i in range(n):
        for j in range(m):
            terminal_weights.append([
                teta_unary[i, j, 0],
                teta_unary[i, j, 1], 
            ])
                
    edge_weights = np.array(edge_weights, dtype=float)
    terminal_weights = np.array(terminal_weights, dtype=float)

    (cut, labels) = graph_cut.graph_cut(terminal_weights, edge_weights)
    return labels, cut + delta_energy

def apply_step(mask, labels, alpha):
    n, m = mask.shape
    for i in range(n):
        for j in range(m):
            s = vertex_id(i, j, n, m)
            if labels[s] == 0:
                mask[i][j] = alpha
    return mask

def alphaExpansionGridPotts(unary, vertC, horC, metric, maxIter=500, display=False, numStart=1, randOrder=False):
    import time
    n, m, k = unary.shape
    best_labels = np.zeros((n, m))
    best_energy = None
    energies = []
    times = []
    start_time = time.time()
    
    for start_id in range(numStart):
        if display:
            print 'start #', start_id
        mask = np.random.choice(k, (n, m))
        
        labels_order = list(range(k))
        if randOrder:
            random.shuffle(labels_order)
        
        old_energy = None
            
        for it in range(maxIter):
            times.append(time.time() - start_time)
            alpha = labels_order[it % len(labels_order)]
            labels, energy = doAlphaExpansion(alpha, mask, n, m, unary, vertC, horC, metric)
            mask = apply_step(mask, labels, alpha)
            energies.append(energy)
            
            if display:
                print '       it:', it
                print '    alpha:', alpha
                print '   energy:', energy
                print
            
            if not old_energy is None and np.abs(old_energy  - energy) < 0.001:
                break
            old_energy = energy
            
        if best_energy is None or old_energy < best_energy:
            best_labels = mask.copy()
            best_energy = old_energy
    return best_labels, energies, times


# In[30]:

def stichImages(images, seeds):
    metric = get_metric(len(images))
    unary = get_unary(images, seeds)
    vertC, horC = get_C(images, seeds)
    resultMask, energy, time = alphaExpansionGridPotts(unary, vertC, horC, metric,  display=True)
    resultImage = images[0].copy()
    n, m, _ = resultImage.shape
    for i in range(n):
        for j in range(m):
            resultImage[i][j] = images[resultMask[i][j]][i][j]
    return resultImage, resultMask

# images= list()
# images.append(np.load("image1.jpg"))
# images.append(np.load("image2.jpg"))
# images.append(np.load("image3.jpg"))
# images.append(np.load("image1.jpg"))
# images.append(np.load("image2.jpg"))
# images.append(np.load("image3.jpg"))
# resImg, resMask = stichImages(images, 
binarizated = np.load("mask3.npy")
plt.imshow(binarizated, interpolation='none', cmap=plt.cm.gray)
plt.show()