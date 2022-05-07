import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as stats
import seaborn as sns
import numpy as np
import torch
import os

def get_neighbors(arr, x, y, kernel_size, available_pixel):
    if kernel_size % 2 == 0:
        raise ValueError('Kernel size must be an odd value')

    half = kernel_size // 2
    
    available_pixel = available_pixel[max(0, x - half):min(x + kernel_size - half, arr.shape[0]), max(0, y - half):min(y + kernel_size - half, arr.shape[1])]
    arr = arr[max(0, x - half):min(x + kernel_size - half, arr.shape[0]), max(0, y - half):min(y + kernel_size - half, arr.shape[1])]

    return arr[np.where(available_pixel == 1)]

def draw(arr, title, save_path = "./out/", cmap="viridis", vmin = None, vmax = None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig, ax = plt.subplots(figsize=(30, 25))
    ax.grid(False)
    ax.set_title(title)
    if (vmin != None):
        im = ax.imshow(arr, cmap=cmap, aspect='auto', vmin = vmin, vmax = vmax)
    else:
        im = ax.imshow(arr, cmap=cmap, aspect='auto')
    fig.colorbar(im)
    fig.savefig(save_path + title)
    plt.close(fig)

def draw_scatter_plot(x, y, xlabel, ylabel, title, save_path, color = "red"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    g = sns.JointGrid(x, y, height=10)
    g = g.plot_joint(plt.scatter, color=color, s=30, edgecolor="white")
    try:
        g = g.plot_marginals(sns.distplot, kde=True, color=color)
    except Exception as e:
        print("Encountered error when plotting join plot. Error: " + str(e))
        print("Predicted values: ", y)
        # g = g.plot_marginals(sns.distplot, kde=False, color=color)

    g = g.annotate(stats.pearsonr)
    g.set_axis_labels(xlabel=xlabel, ylabel=ylabel)

    g.ax_joint.plot([0, 1], [0, 1], c='r', linestyle='--', alpha=0.9, label='Ideal prediction')
    g.savefig('{}{}.png'.format(save_path, title))
    plt.close()

def extract_categorical(row, col, save_dir, data_dir):
    ## Categorical
    all_pred1 = torch.load('{}/all_pred1.pt'.format(save_dir))

    categorical_tasks = all_pred1['all_target_cls'].shape[1]
    all_pred_cls = np.zeros((row, col, categorical_tasks))
    all_target_cls = np.zeros((row, col, categorical_tasks))
    available_pixel = np.zeros((row, col))

    complete_image1 = np.load('{}/complete_image1.npy'.format(data_dir), allow_pickle=True)
    print('start complete_image 1')
    for i in range(all_pred1['all_target_cls'].shape[0]):
        r, c, _, _, _ = complete_image1[i]
        all_pred_cls[r, c] = all_pred1['all_pred_cls'][i]
        all_target_cls[r, c] = all_pred1['all_target_cls'][i]
        available_pixel[r, c] = 1
        if i % 100000 == 0:
            print(i)

    all_pred2 = torch.load('{}/all_pred2.pt'.format(save_dir))
    complete_image2 = np.load('{}/complete_image2.npy'.format(data_dir), allow_pickle=True)
    print('start complete_image 2')
    for i in range(all_pred2['all_target_cls'].shape[0]):
        r, c, _, _, _ = complete_image2[i]
        all_pred_cls[r, c] = all_pred2['all_pred_cls'][i]
        all_target_cls[r, c] = all_pred2['all_target_cls'][i]
        available_pixel[r, c] = 1
        if i % 100000 == 0:
            print(i)

    all_pred3 = torch.load('{}/all_pred3.pt'.format(save_dir))
    complete_image3 = np.load('{}/complete_image3.npy'.format(data_dir), allow_pickle=True)
    print('start complete_image 3')
    for i in range(all_pred3['all_target_cls'].shape[0]):
        r, c, _, _, _ = complete_image3[i]
        all_pred_cls[r, c] = all_pred3['all_pred_cls'][i]
        all_target_cls[r, c] = all_pred3['all_target_cls'][i]
        available_pixel[r, c] = 1
        if i % 100000 == 0:
            print(i)

    for t in range(categorical_tasks):
        print('index', t)
        pred = all_pred_cls[:, :, t]
        target = all_target_cls[:, :, t]
        print('start that {}'.format(t))
        np.save('{}/pred_cls_{}.npy'.format(save_dir, t), pred)
        np.save('{}/target_cls_{}.npy'.format(save_dir, t), target)

def extract_regression(row, col, save_dir, data_dir):
    ## Regression
    all_pred1 = torch.load('{}/complete_image1.pt'.format(save_dir))

    regression_tasks = all_pred1['all_target_reg'].shape[1]
    all_pred_reg = np.zeros((row, col, regression_tasks))
    all_target_reg = np.zeros((row, col, regression_tasks))
    available_pixel = np.zeros((row, col))

    complete_image1 = np.load('{}/complete_image1.npy'.format(data_dir), allow_pickle=True)
    print('start complete_image 1')
    for i in range(all_pred1['all_target_reg'].shape[0]):
        r, c, _, _, _ = complete_image1[i]
        all_pred_reg[r, c] = all_pred1['all_pred_reg'][i]
        all_target_reg[r, c] = all_pred1['all_target_reg'][i]
        available_pixel[r, c] = 1
        if i % 100000 == 0:
            print(i)

    all_pred2 = torch.load('{}/complete_image2.pt'.format(save_dir))
    complete_image2 = np.load('{}/complete_image2.npy'.format(data_dir), allow_pickle=True)
    print('start complete_image 2')
    for i in range(all_pred2['all_target_reg'].shape[0]):
        r, c, _, _, _ = complete_image2[i]
        all_pred_reg[r, c] = all_pred2['all_pred_reg'][i]
        all_target_reg[r, c] = all_pred2['all_target_reg'][i]
        available_pixel[r, c] = 1
        if i % 100000 == 0:
            print(i)

    all_pred3 = torch.load('{}/complete_image3.pt'.format(save_dir))
    complete_image3 = np.load('{}/complete_image3.npy'.format(data_dir), allow_pickle=True)
    print('start complete_image 3')
    for i in range(all_pred3['all_target_reg'].shape[0]):
        r, c, _, _, _ = complete_image3[i]
        all_pred_reg[r, c] = all_pred3['all_pred_reg'][i]
        all_target_reg[r, c] = all_pred3['all_target_reg'][i]
        available_pixel[r, c] = 1
        if i % 100000 == 0:
            print(i)

    for t in range(regression_tasks):
        print('index', t)
        pred = all_pred_reg[:, :, t]
        target = all_target_reg[:, :, t]
        std = np.zeros((row, col))
        print('start that {}'.format(t))

        for i in range(all_pred1['all_target_reg'].shape[0]):
            r, c, _, _, _ = complete_image1[i]
            neighbors = get_neighbors(pred, r, c, 5, available_pixel)
            # variance[r, c] = neighbors.var()
            std[r, c] = neighbors.std()
            if i % 100000 == 0:
                print(i)

        for i in range(all_pred2['all_target_reg'].shape[0]):
            r, c, _, _, _ = complete_image2[i]
            neighbors = get_neighbors(pred, r, c, 5, available_pixel)
            # variance[r, c] = neighbors.var()
            std[r, c] = neighbors.std()
            if i % 100000 == 0:
                print(i)

        for i in range(all_pred3['all_target_reg'].shape[0]):
            r, c, _, _, _ = complete_image3[i]
            neighbors = get_neighbors(pred, r, c, 5, available_pixel)
            # variance[r, c] = neighbors.var()
            std[r, c] = neighbors.std()
            if i % 100000 == 0:
                print(i)

        np.save('{}/pred_reg_{}.npy'.format(save_dir, t), pred)
        np.save('{}/target_reg_{}.npy'.format(save_dir, t), target)
        np.save('{}/std5x5_reg_{}.npy'.format(save_dir, t), std)

def new_TAIGA_extract_regression(row, col, save_dir, data_dir):
    ## Regression
    all_pred1 = torch.load('{}/complete_image1.pt'.format(save_dir))

    regression_tasks = all_pred1['all_target_reg'].shape[1]
    all_pred_reg = np.zeros((row, col, regression_tasks))
    all_target_reg = np.zeros((row, col, regression_tasks))
    available_pixel = np.zeros((row, col))

    complete_image1 = np.load('{}/complete_image1.npy'.format(data_dir), allow_pickle=True)
    print('start complete_image 1')
    for i in range(all_pred1['all_target_reg'].shape[0]):
        r, c, _, _, _ = complete_image1[i]
        all_pred_reg[r, c] = all_pred1['all_pred_reg'][i]
        all_target_reg[r, c] = all_pred1['all_target_reg'][i]
        available_pixel[r, c] = 1
        if i % 100000 == 0:
            print(i)

    all_pred2 = torch.load('{}/complete_image2.pt'.format(save_dir))
    complete_image2 = np.load('{}/complete_image2.npy'.format(data_dir), allow_pickle=True)
    print('start complete_image 2')
    for i in range(all_pred2['all_target_reg'].shape[0]):
        r, c, _, _, _ = complete_image2[i]
        all_pred_reg[r, c] = all_pred2['all_pred_reg'][i]
        all_target_reg[r, c] = all_pred2['all_target_reg'][i]
        available_pixel[r, c] = 1
        if i % 100000 == 0:
            print(i)

    all_pred3 = torch.load('{}/complete_image3.pt'.format(save_dir))
    complete_image3 = np.load('{}/complete_image3.npy'.format(data_dir), allow_pickle=True)
    print('start complete_image 3')
    for i in range(all_pred3['all_target_reg'].shape[0]):
        r, c, _, _, _ = complete_image3[i]
        all_pred_reg[r, c] = all_pred3['all_pred_reg'][i]
        all_target_reg[r, c] = all_pred3['all_target_reg'][i]
        available_pixel[r, c] = 1
        if i % 100000 == 0:
            print(i)


    all_pred_reg2 = np.zeros((row, col, regression_tasks))
    all_target_reg2 = np.zeros((row, col, regression_tasks))
    new_complete_image = np.load('{}/complete_image.npy'.format('../data/new_TAIGA'), allow_pickle=True)
    print('start complete_image 4')
    for j in range(new_complete_image.shape[0]):
        r, c, _, _, _ = new_complete_image[j]
        all_pred_reg2[r, c] = all_pred_reg[r, c]
        all_target_reg2[r, c] = all_target_reg[r, c]
        if j % 100000 == 0:
            print(j)


    for t in range(regression_tasks):
        print('index', t)
        pred = all_pred_reg2[:, :, t]
        target = all_target_reg2[:, :, t]
        std = np.zeros((row, col))
        print('start that {}'.format(t))

        # for i in range(all_pred1['all_target_reg'].shape[0]):
        #     r, c, _, _, _ = complete_image1[i]
        #     neighbors = get_neighbors(pred, r, c, 5, available_pixel)
        #     # variance[r, c] = neighbors.var()
        #     std[r, c] = neighbors.std()
        #     if i % 100000 == 0:
        #         print(i)

        # for i in range(all_pred2['all_target_reg'].shape[0]):
        #     r, c, _, _, _ = complete_image2[i]
        #     neighbors = get_neighbors(pred, r, c, 5, available_pixel)
        #     # variance[r, c] = neighbors.var()
        #     std[r, c] = neighbors.std()
        #     if i % 100000 == 0:
        #         print(i)

        # for i in range(all_pred3['all_target_reg'].shape[0]):
        #     r, c, _, _, _ = complete_image3[i]
        #     neighbors = get_neighbors(pred, r, c, 5, available_pixel)
        #     # variance[r, c] = neighbors.var()
        #     std[r, c] = neighbors.std()
        #     if i % 100000 == 0:
        #         print(i)

        np.save('{}/new_TAIGA_pred_reg_{}.npy'.format(save_dir, t), pred)
        np.save('{}/new_TAIGA_target_reg_{}.npy'.format(save_dir, t), target)
        # np.save('{}/new_TAIGA_std5x5_reg_{}.npy'.format(save_dir, t), std)


def extract_stand_regression(row, col, save_dir, data_dir):
    all_pred = torch.load('{}/stand_all_pred_reg.pt'.format(save_dir))
    all_target = torch.load('{}/stand_all_target_reg.pt'.format(save_dir))
    all_std = torch.load('{}/stand_all_std_reg.pt'.format(save_dir))
    regression_tasks = all_pred.shape[1]

    all_pred_reg = np.zeros((row, col, regression_tasks))
    all_target_reg = np.zeros((row, col, regression_tasks))
    all_std_reg = np.zeros((row, col, regression_tasks))

    complete_image = np.load('{}/complete_image.npy'.format(data_dir), allow_pickle=True)

    print('start complete_image')
    for i in range(all_pred.shape[0]):
        r, c, _, _, _ = complete_image[i]
        all_pred_reg[r, c] = all_pred[i]
        all_target_reg[r, c] = all_target[i]
        all_std_reg[r, c] = all_std[i]
        if i % 100000 == 0:
            print(i)

    for t in range(regression_tasks):
        print('index', t)
        pred = all_pred_reg[:, :, t]
        target = all_target_reg[:, :, t]
        std = all_std_reg[:, :, t]
        np.save('{}/stand_pred_reg_{}.npy'.format(save_dir, t), pred)
        np.save('{}/stand_target_reg_{}.npy'.format(save_dir, t), target)
        np.save('{}/stand_std_reg_{}.npy'.format(save_dir, t), std)

def new_TAIGA_extract_stand_regression(row, col, save_dir, data_dir):
    all_pred = torch.load('{}/stand_all_pred_reg.pt'.format(save_dir))
    all_target = torch.load('{}/stand_all_target_reg.pt'.format(save_dir))
    all_std = torch.load('{}/stand_all_std_reg.pt'.format(save_dir))
    regression_tasks = all_pred.shape[1]

    all_pred_reg = np.zeros((row, col, regression_tasks))
    all_target_reg = np.zeros((row, col, regression_tasks))
    all_std_reg = np.zeros((row, col, regression_tasks))

    complete_image = np.load('{}/complete_image.npy'.format(data_dir), allow_pickle=True)

    print('start complete_image')
    for i in range(all_pred.shape[0]):
        r, c, _, _, _ = complete_image[i]
        all_pred_reg[r, c] = all_pred[i]
        all_target_reg[r, c] = all_target[i]
        all_std_reg[r, c] = all_std[i]
        if i % 100000 == 0:
            print(i)


    all_pred_reg2 = np.zeros((row, col, regression_tasks))
    all_target_reg2 = np.zeros((row, col, regression_tasks))
    all_std_reg2 = np.zeros((row, col, regression_tasks))
    new_complete_image = np.load('{}/complete_image.npy'.format('../data/new_TAIGA'), allow_pickle=True)
    print('start complete_image 2')
    for j in range(new_complete_image.shape[0]):
        r, c, _, _, _ = new_complete_image[j]
        all_pred_reg2[r, c] = all_pred_reg[r, c]
        all_target_reg2[r, c] = all_target_reg[r, c]
        all_std_reg2[r, c] = all_std_reg[r, c]
        if j % 100000 == 0:
            print(j)

    for t in range(regression_tasks):
        print('index', t)
        pred = all_pred_reg[:, :, t]
        target = all_target_reg[:, :, t]
        std = all_std_reg[:, :, t]
        np.save('{}/new_TAIGA_stand_pred_reg_{}.npy'.format(save_dir, t), pred)
        np.save('{}/new_TAIGA_stand_target_reg_{}.npy'.format(save_dir, t), target)
        np.save('{}/new_TAIGA_stand_std_reg_{}.npy'.format(save_dir, t), std)


def extract_stand_categorical(row, col, save_dir, data_dir):
    all_pred = torch.load('{}/stand_all_pred_cls.pt'.format(save_dir)) + 1
    all_target = torch.load('{}/stand_all_target_cls.pt'.format(save_dir)) + 1
    categorical_tasks = all_pred.shape[1]

    all_pred_cls = np.zeros((row, col, categorical_tasks))
    all_target_cls = np.zeros((row, col, categorical_tasks))

    complete_image = np.load('{}/complete_image.npy'.format(data_dir), allow_pickle=True)
    print('start complete_image')
    for i in range(all_pred.shape[0]):
        r, c, _, _, _ = complete_image[i]
        all_pred_cls[r, c] = all_pred[i]
        all_target_cls[r, c] = all_target[i]
        if i % 100000 == 0:
            print(i)

    for t in range(categorical_tasks):
        print('index', t)
        pred = all_pred_cls[:, :, t]
        target = all_target_cls[:, :, t]
        np.save('{}/stand_pred_cls_{}.npy'.format(save_dir, t), pred)
        np.save('{}/stand_target_cls_{}.npy'.format(save_dir, t), target)



def main():
    save_dir = './FL_Pham34-12012021-gn-aug-eelis-patchsize45/'
    data_dir = '../data/TAIGA'
    metadata = torch.load('{}/metadata.pt'.format(data_dir))
    label_names = metadata['reg_label_names']

    
    
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, 15)]

    # row = max(complete_image[:, 0])
    # col = max(complete_image[:, 1])
    row = 12143
    col = 12826
    
    

    # extract_categorical(row, col, save_dir, data_dir)
    # extract_regression(row, col, save_dir, data_dir)
    # extract_stand_regression(row, col, save_dir, data_dir)
    # new_TAIGA_extract_stand_regression(row, col, save_dir, data_dir)
    new_TAIGA_extract_regression(row, col, save_dir, data_dir)
    # extract_stand_categorical(row, col, save_dir, data_dir)
    
if __name__ == "__main__":
    main()