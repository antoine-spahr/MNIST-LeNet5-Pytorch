import matplotlib
import matplotlib.pyplot as plt
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

def show_samples(images, labels, pred, seed=1):
    """
    Show a sample of correctly and incorrectly classified samples.
    ----------
    INPUT
        |---- images (torch.)
    """
    correct_mask = pred == labels
    correct_image, correct_label, correct_pred = images[correct_mask, :, :], labels[correct_mask], pred[correct_mask]
    incorrect_mask = pred != labels
    incorrect_image, incorrect_label, incorrect_pred = images[incorrect_mask, :, :], labels[incorrect_mask], pred[incorrect_mask]

    n1, n2 = 5, 10
    fig, axs = plt.subplots(n1, n2, figsize=(18, 9), gridspec_kw={'hspace':0.2, 'wspace':0.01})

    # get correctly classified sample
    prng = np.random.RandomState(seed=1)
    ind = prng.randint(low=0, high=correct_image.shape[0], size=(n1, n2//2))
    for i in range(n1):
        for j in range(n2//2):
            axs[i,j].imshow(correct_image[ind[i,j]], cmap='gray')
            axs[i,j].set_axis_off()
            title = axs[i,j].set_title(f'True {correct_label[ind[i,j]]} ; Pred {correct_pred[ind[i,j]]}', color='limegreen', fontweight='bold', fontsize=10)

    # get incorrectly classified sample
    prng = np.random.RandomState(seed=seed)
    ind = prng.randint(low=0, high=incorrect_image.shape[0], size=(n1, n2//2))
    for i in range(n1):
        for j in range(n2//2):
            axs[i,j + n2//2].imshow(incorrect_image[ind[i,j]], cmap='gray')
            axs[i,j + n2//2].set_axis_off()
            title = axs[i,j + n2//2].set_title(f'True {incorrect_label[ind[i,j]]} ; Pred {incorrect_pred[ind[i,j]]}', color='crimson', fontweight='bold', fontsize=10)
