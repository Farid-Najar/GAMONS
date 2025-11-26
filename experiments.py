import numpy as np
import torch

from utils import  prox_mcp, mcp, mcp_torch
from BCD import load_dataset

from problems import run_MoreauNSD, run_VS, run_subgradient_descent

import seaborn as sns
sns.set_theme('paper', 'whitegrid')
import matplotlib.pyplot as plt

def run_experiment(
    dataset_name: str = 'camera',
    rank: int = 10,
    K: int = 5_000,
    g: callable = lambda W : np.sum(mcp(W)),
    g_torch: callable = lambda W : torch.sum(mcp_torch(W)),
    prox: callable = prox_mcp,
    store_WH_every: int = 1,
    ):
    """
    Run an experiment on a given dataset.

    Parameters
    ----------
    dataset_name : str, optional
        The name of the dataset to use, by default 'camera'
        Possible dataset : 
            'synthetic', 'olivetti', 'camera', 'spectrometer', 'football', 'miserables', 'low_rank_synthetic'
    rank : int, optional
        The rank of the low-rank approximation, by default 10
    K : int, optional
        The number of iterations to run, by default 5_000
    """
    D = load_dataset(dataset_name)
    
    results = {}
    results['K'] = K
    results['rank'] = rank

    # Min-Max Scaling
    min_vals = D.min()
    max_vals = D.max()

    D = (D - min_vals) / (max_vals - min_vals)
    results['original'] = D
    m, n = D.shape
    norm_D = np.linalg.norm(D, 'fro')**2
    results['norm'] = norm_D
    F_min = g(D)
    results['F_min'] = F_min


    # Running the experiments
    loss_NSD, dist_W_prox_NSD, WHs_NSD = run_MoreauNSD(
        D, rank, prox, max_iter = K, store_WH_every = store_WH_every
    )
    print(f'Gamons (p = 2/3. q = 1/4) loss: {loss_NSD[-1]}')
    results['gamons (p = 2/3, q = 1/4)'] = {
        'loss': loss_NSD,
        'dist_W_prox': dist_W_prox_NSD,
        'WHs': WHs_NSD,
    }
    loss_NSD2, dist_W_prox_NSD2, WHs_NSD2 = run_MoreauNSD(
        D, rank, prox, max_iter = K, p = 7/12, q = 1/3, store_WH_every = store_WH_every
    )
    print(f'Gamons (p = 7/12. q = 1/3) loss: {loss_NSD2[-1]}')
    results['gamons (p = 7/12, q = 1/3)'] = {
        'loss': loss_NSD2,
        'dist_W_prox': dist_W_prox_NSD2,
        'WHs': WHs_NSD2,
    }
    # loss_NSD3, dist_W_prox_NSD3, WHs_NSD3 = run_MoreauNSD(D, rank, prox, max_iter = K, p = 1/2, q = 1/2)  
    # loss_NSD3, dist_W_prox_NSD3, WHs_NSD3 = run_MoreauNSD(D, rank, prox, max_iter = K, p = 1/2, q = 1/2)
    # loss_NSD4, dist_W_prox_NSD4, WHs_NSD4 = run_MoreauNSD(D, rank, prox, max_iter = K, p = 1/3, q = 1/2)
    loss_sub, _, WHs_sub = run_subgradient_descent(
        D, g_torch,((m, rank), (rank, n)), max_iter = K, step_size_rule = 5e-4, 
        store_WH_every = store_WH_every
    ) 
    print(f'Subgradient descent loss: {loss_sub[-1]}')
    results['subgradient'] = {
        'loss': loss_sub,
        'WHs': WHs_sub,
    }
    # loss_cvxNSD, dist_W_prox_cvxNSD, WHs_cvxNSD = run_cvxMoreauNSD(D, rank, prox, max_iter = K)

    # plt.scatter(np.arange(len(loss))[::50], loss[::50]/norm_D, label = 'GAMONS', marker="v")

    # loss = run_MoreauNSD(D, 10, lmo = lmo_fro)
    # plt.loglog(loss/norm_D, label = 'l2 lmo')

    loss_VS, dist_W_prox_VS, WHs_VS = run_VS(
        D, rank, prox, max_iter = K, store_WH_every = store_WH_every
    )
    print(f'Variable Smoothing BW loss: {loss_VS[-1]}')
    results['variable smoothing'] = {
        'loss': loss_VS,
        'dist_W_prox': dist_W_prox_VS,
        'WHs': WHs_VS,
    }
    
    return results
    

def plot_scaling(dataset_name: str = 'camera'):
    D = load_dataset(dataset_name)
    img = plt.imshow(D)
    plt.title('Original')
    img.set_cmap('gray')
    plt.axis('off')
    plt.show()

    # Min-Max Scaling
    min_vals = D.min()
    max_vals = D.max()

    D = (D - min_vals) / (max_vals - min_vals)

    plt.title('After Min-Max scaling')
    img = plt.imshow(D)
    img.set_cmap('gray')
    plt.axis('off')
    plt.show()

    # # Normalization
    # D /= np.linalg.norm(D, 'fro')
    # plt.title('After normalization')
    # plt.imshow(D)
    # plt.show()

    print(f'Dataset shape: {D.shape}')
    
def plot_images(results: dict):
    D = results['original']
    norm_D = results['norm']
    
    # Plotting the results
    img = plt.imshow(D)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('Original')
    plt.show()

    W, H = results['variable smoothing']['WHs'][-1]
    img = plt.imshow(W@H)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('VS')
    plt.show()
    
    W, H = results['gamons (p = 2/3, q = 1/4)']['WHs'][-1]
    img = plt.imshow(norm_D*W@H)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('Gamons (p = 2/3, q = 1/4)')
    plt.show()

    
    # W, H = WHs_cvxNSD[-1]
    # ax[3].imshow(norm_D*W@H)
    # ax[3].set_title('Ours (CVX MNSD)')
    W, H = results['gamons (p = 7/12, q = 1/3)']['WHs'][-1]
    img = plt.imshow(norm_D*W@H)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('Gamons (p = 7/12, q = 1/3)')
    plt.show()

    W, H = results['subgradient']['WHs'][-1]
    img = plt.imshow(norm_D*W@H)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('Subgradient')
    plt.show()
    
def plot_loss(results: dict):
    F_min = results['F_min']
    K = results['K']
    scatter_period = K // 20
    norm_D = results['norm']

    x = np.arange(len(results['variable smoothing']['loss']))[::scatter_period]
    x = np.logspace(0, np.log10(len(results['variable smoothing']['loss'])-1), num=len(x), endpoint=True).astype(int)
    
    plt.loglog(results['variable smoothing']['loss']/norm_D)
    plt.scatter(x, results['variable smoothing']['loss'][x]/norm_D, label = 'Variable Smoothing BW', marker="o")
    print(f"VS : {results['variable smoothing']['loss'][-1]/norm_D:.3e}")
    
    plt.loglog(results['gamons (p = 2/3, q = 1/4)']['loss']/norm_D)
    plt.scatter(x, results['gamons (p = 2/3, q = 1/4)']['loss'][x]/norm_D, label = 'GAMONS (p = 2/3. q = 1/4)', marker="v")
    print(f"GAMONS (p = 2/3. q = 1/4) : {results['gamons (p = 2/3, q = 1/4)']['loss'][-1]/norm_D:.3e}")
    
    plt.loglog(results['gamons (p = 7/12, q = 1/3)']['loss']/norm_D)
    plt.scatter(x, results['gamons (p = 7/12, q = 1/3)']['loss'][x]/norm_D, label = 'GAMONS (p = 7/12. q = 1/3)', marker="s")
    print(f"GAMONS (p = 7/12. q = 1/3) : {results['gamons (p = 7/12, q = 1/3)']['loss'][-1]/norm_D:.3e}")

    plt.loglog(results['subgradient']['loss']/norm_D)
    plt.scatter(x, results['subgradient']['loss'][x]/norm_D, label = 'Subgradient', marker="*")
    print(f"Subgradient : {results['subgradient']['loss'][-1]/norm_D:.3e}")

    plt.ylabel(r'$\frac{\|Y - WH\|_F^2}{\|Y\|_F^2}$')
    plt.xlabel('Iterations')
    plt.title('The reconstruction loss')

    plt.legend()
    plt.show()
    
def plot_smoothing_loss(results: dict):
    F_min = results['F_min']
    K = results['K']
    scatter_period = K // 20
    norm_D = results['norm']

    x = np.arange(len(results['variable smoothing']['dist_W_prox']))[::scatter_period]
    x = np.logspace(0, np.log10(len(results['variable smoothing']['dist_W_prox'])-1), num=len(x), endpoint=True).astype(int)
    
    plt.loglog(results['variable smoothing']['dist_W_prox']/norm_D)
    plt.scatter(x, results['variable smoothing']['dist_W_prox'][x]/norm_D, label = 'Variable Smoothing BW', marker="o")
    print(f"VS : {results['variable smoothing']['dist_W_prox'][-1]/norm_D:.3e}")
    
    plt.loglog(results['gamons (p = 2/3, q = 1/4)']['dist_W_prox']/norm_D)
    plt.scatter(x, results['gamons (p = 2/3, q = 1/4)']['dist_W_prox'][x]/norm_D, label = 'GAMONS (p = 2/3. q = 1/4)', marker="v")
    print(f"GAMONS (p = 2/3. q = 1/4) : {results['gamons (p = 2/3, q = 1/4)']['dist_W_prox'][-1]/norm_D:.3e}")
    
    
    plt.loglog(results['gamons (p = 7/12, q = 1/3)']['dist_W_prox']/norm_D)
    plt.scatter(x, results['gamons (p = 7/12, q = 1/3)']['dist_W_prox'][x]/norm_D, label = 'GAMONS (p = 7/12. q = 1/3)', marker="s")
    print(f"GAMONS (p = 7/12. q = 1/3) : {results['gamons (p = 7/12, q = 1/3)']['dist_W_prox'][-1]/norm_D:.3e}")
    
    plt.ylabel(r'$\|W - prox_{\beta_k}(W)\|$')
    plt.xlabel('Iterations')
    plt.title('The smoothing loss')

    plt.legend()
    plt.show()
    
def plot_primal_gap_and_penalty(
    results: dict, 
    g:callable = lambda W : np.sum(mcp(W)),
    ):
    F_min = results['F_min']
    K = results['K']
    scatter_period = K // 20
    norm_D = results['norm']
    D = results['original']

    x = np.arange(K)[::scatter_period]
    x = np.logspace(0, np.log10(K-1), num=len(x), endpoint=True).astype(int)
    
    ls_VS = np.zeros(K+1)
    g_VS = np.zeros(K+1)
    for i, (W, H) in enumerate(results['variable smoothing']['WHs']):
        g_VS[i] = g(W)
        ls_VS[i] = np.linalg.norm(D - W@H, 'fro')**2 + g_VS[i]
    plt.loglog(ls_VS)
    plt.scatter(x, ls_VS[x], label = 'Variable Smoothing BW', marker="o")
    print(f"VS : {ls_VS[-1]:.3e}")
    
    ls_NSD = np.zeros(K+1)
    g_NSD = np.zeros(K+1)
    for i, (W, H) in enumerate(results['gamons (p = 2/3, q = 1/4)']['WHs']):
        g_NSD[i] = g(W)
        ls_NSD[i] = np.linalg.norm(D - W@H, 'fro')**2 + g_NSD[i]
    plt.loglog(ls_NSD)  
    plt.scatter(x, ls_NSD[x], label = 'GAMONS (p = 2/3. q = 1/4)', marker="v")
    print(f"GAMONS (p = 2/3. q = 1/4) : {ls_NSD[-1]:.3e}")
    
    ls_NSD2 = np.zeros(K+1)
    g_NSD2 = np.zeros(K+1)
    for i, (W, H) in enumerate(results['gamons (p = 7/12, q = 1/3)']['WHs']):
        g_NSD2[i] = g(W)
        ls_NSD2[i] = np.linalg.norm(D - W@H, 'fro')**2 + g_NSD2[i]
    plt.loglog(ls_NSD2)  
    plt.scatter(x, ls_NSD2[x], label = 'GAMONS (p = 7/12. q = 1/3)', marker="s")
    print(f"GAMONS (p = 7/12. q = 1/3) : {ls_NSD2[-1]:.3e}")
    
    ls_sub = np.zeros(K+1)
    g_sub = np.zeros(K+1)
    for i, (W, H) in enumerate(results['subgradient']['WHs']):
        g_sub[i] = g(W)
        ls_sub[i] = np.linalg.norm(D - W@H, 'fro')**2 + g_sub[i]
    plt.loglog(ls_sub)
    plt.scatter(x, ls_sub[x], label = 'Subgradient', marker="*")
    print(f"Subgradient : {ls_sub[-1]:.3e}")
    
    plt.ylabel(r'$F(x_k)$')
    plt.xlabel(r'Iterations $k$')
    plt.title(r'Objectice function $F(x_k)$')

    plt.legend()
    plt.show()
    
    plt.loglog(g_VS)
    plt.scatter(x, g_VS[x], label = 'Variable Smoothing BW', marker="o")
    print(f"VS : {g_VS[-1]:.3e}")
    
    plt.loglog(g_NSD)
    plt.scatter(x, g_NSD[x], label = 'GAMONS (p = 2/3. q = 1/4)', marker="v")
    print(f"GAMONS (p = 2/3. q = 1/4) : {g_NSD[-1]:.3e}")
    
    plt.loglog(g_NSD2)
    plt.scatter(x, g_NSD2[x], label = 'GAMONS (p = 7/12. q = 1/3)', marker="s")
    print(f"GAMONS (p = 7/12. q = 1/3) : {g_NSD2[-1]:.3e}")
    
    plt.loglog(g_sub)
    plt.scatter(x, g_sub[x], label = 'Subgradient', marker="*")
    plt.ylabel(r'$g(W_k)$')
    plt.xlabel('Iterations')
    plt.title('The penalty')

    plt.legend()
    plt.show()

def plot_images_ranks(results: dict):
    for rank in results.keys():
        fig, ax = plt.subplots(1, 5, figsize=(16, 4))
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        ax[3].axis('off')
        ax[4].axis('off')
        
        plt.suptitle(f'rank = {rank}', x=0.05, y=0.5, ha='left', va='center', fontsize=12)
        
        ax[0].imshow(results[rank]['original'])
        ax[0].set_title('Original')
        
        W, H = results[rank]['variable smoothing']['WHs'][-1]
        ax[1].imshow(W@H)
        ax[1].set_title('Variable Smoothing BW')
        W, H = results[rank]['gamons (p = 2/3, q = 1/4)']['WHs'][-1]
        ax[2].imshow(W@H)
        ax[2].set_title('GAMONS (p = 2/3. q = 1/4)')
        W, H = results[rank]['gamons (p = 7/12, q = 1/3)']['WHs'][-1]
        ax[3].imshow(W@H)
        ax[3].set_title('GAMONS (p = 7/12. q = 1/3)')
        W, H = results[rank]['subgradient']['WHs'][-1]
        ax[4].imshow(W@H)
        ax[4].set_title('Subgradient')
        
        plt.show()
