import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', family='Hiragino Sans GB')
import numpy as np

def print_progessbar(N, Max, Name='', Size=10, erase=False):
    """
    Print a progress bar. To be used in a for-loop and called at each iteration
    with the iteration number and the max number of iteration.
    ------------
    INPUT
        |---- N (int) the iteration current number
        |---- Max (int) the total number of iteration
        |---- Name (str) an optional name for the progress bar
        |---- Size (int) the size of the progress bar
        |---- erase (bool) whether to erase the progress bar when 100% is reached.
    OUTPUT
        |---- None
    """
    print(f'{Name} {N+1:04d}/{Max:04d}'.ljust(len(Name) + 12) \
        + f'|{"█"*int(Size*(N+1)/Max)}'.ljust(Size+1) + f'| {(N+1)/Max:.1%}'.ljust(6), \
        end='\r')

    if N+1 == Max:
        if erase:
            print(' '.ljust(len(Name) + Size + 40), end='\r')
        else:
            print('')

def show_samples(idx_label_pred, dataset, seed=-1, save_path='', n=(5, 10), lut=None):
    """
    Show a sample of correctly and incorrectly classified samples.
    """
    idx, labels, pred = idx_label_pred
    idx, labels, pred = np.array(idx), np.array(labels), np.array(pred)
    # get correct and incrorect
    correct_mask = pred == labels
    correct_idx, correct_label, correct_pred = idx[correct_mask], labels[correct_mask], pred[correct_mask]
    incorrect_mask = pred != labels
    incorrect_idx, incorrect_label, incorrect_pred = idx[incorrect_mask], labels[incorrect_mask], pred[incorrect_mask]

    n1, n2 = n
    fig, axs = plt.subplots(n1, n2, figsize=(n2*2, n1*1.2*2), gridspec_kw={'hspace':0.4, 'wspace':0.01})

    # get correctly classified sample
    prng = np.random.RandomState(seed=seed) if seed != -1 else np.random
    ind = prng.randint(low=0, high=correct_idx.shape[0], size=(n1, n2//2))

    for i in range(n1):
        for j in range(n2//2):
            axs[i,j].imshow(dataset[correct_idx[ind[i,j]]][0][0,:,:].numpy(), cmap='gray')
            axs[i,j].set_axis_off()
            if lut:
                cor_label = lut[correct_label[ind[i,j]]]
                cor_pred = lut[correct_pred[ind[i,j]]]
            else:
                cor_label = correct_label[ind[i,j]]
                cor_pred = correct_pred[ind[i,j]]
            title = axs[i,j].set_title(f'True: {cor_label}\nPred: {cor_pred}', color='limegreen', fontweight='bold', fontsize=10)

    # get incorrectly classified sample
    prng = np.random.RandomState(seed=seed) if seed != -1 else np.random
    ind = prng.randint(low=0, high=incorrect_idx.shape[0], size=(n1, n2//2))

    for i in range(n1):
        for j in range(n2//2):
            axs[i,j + n2//2].imshow(dataset[incorrect_idx[ind[i,j]]][0][0,:,:].numpy(), cmap='gray')
            axs[i,j + n2//2].set_axis_off()
            if lut:
                incor_label = lut[incorrect_label[ind[i,j]]]
                incor_pred = lut[incorrect_pred[ind[i,j]]]
            else:
                incor_label = incorrect_label[ind[i,j]]
                incor_pred = incorrect_pred[ind[i,j]]
            title = axs[i,j + n2//2].set_title(f'True: {incor_label}\nPred: {incor_pred}', color='crimson', fontweight='bold', fontsize=10)

    fig.savefig(save_path, dpi=200, bbox_inches='tight')
