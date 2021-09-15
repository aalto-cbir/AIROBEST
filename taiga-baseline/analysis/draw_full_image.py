import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib
import scipy.stats as stats
import seaborn as sns
import numpy as np
import torch
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def draw_scatter_plot(x, y, xlabel, ylabel, title, save_path, color = "red", set_xlim=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # g = sns.JointGrid(x, y, height=10)
    # g = g.plot_joint(plt.scatter, color=color, s=30, edgecolor="white")
    # try:
    #     g = g.plot_marginals(sns.distplot, kde=True, color=color)
    # except Exception as e:
    #     print("Encountered error when plotting join plot. Error: " + str(e))
    #     print("Predicted values: ", y)
        # g = g.plot_marginals(sns.distplot, kde=False, color=color)
    
    # g = g.annotate(stats.pearsonr)
    # g.set_axis_labels(xlabel=xlabel, ylabel=ylabel)    
    # g.ax_joint.plot([0, 1], [0, 1], c='r', linestyle='--', alpha=0.9, label='Ideal prediction')
    # g.savefig('{}{}.png'.format(save_path, title))
    
    # hexbin plot
    
    # g2 = sns.jointplot(x, y, height=10, kind='kde', color=color, fill=True, thresh=0.01, levels=10)
    # g2 = sns.jointplot(x, y, height=10, kind='scatter', color=color)
    # vmin = min(min(x), min(y))
    # vmax = max(max(x), max(y))
    # buffer = vmax * 0.1
    vmin = min(min(x), min(y))
    vmax = max(max(x), max(y))
    buffer = vmax * 0.1
    print(min(x), min(y))
    print(max(x), max(y))

    
    if set_xlim:        
        xlimit=(vmin - buffer, vmax + buffer)
        ylimit=(vmin - buffer, vmax + buffer)
    else:
        xlimit=None
        ylimit=None
    g2 = sns.JointGrid(x, y, height=10, xlim=xlimit, ylim=ylimit)

    g2 = g2.plot_joint(sns.kdeplot, color=color, fill=True, thresh=0.05, levels=10)
    g2 = g2.plot_joint(plt.scatter, color=color, s=30, edgecolor="white", alpha=0.3)
    g2.plot_marginals(sns.distplot, kde=True, color=color)
    # g2 = g2.annotate(stats.pearsonr)
    g2.set_axis_labels(xlabel=xlabel, ylabel=ylabel)
    
    # g2.ax_joint.plot(c='r', linestyle='--', alpha=0.9, label='Ideal prediction')
    if set_xlim:
        g2.ax_joint.plot([vmin, vmax], [vmin, vmax], c='r', linestyle='--', alpha=0.9, label='Ideal prediction')
    g2.savefig('{}kde_{}.png'.format(save_path, title))
    plt.close()

    # vmin = min(min(x), min(y))
    # vmax = max(max(x), max(y))
    # buffer = vmax * 0.1
    # g3 = sns.jointplot(x, y, height=10, kind='kde', color=color, fill=True, thresh=0.01, levels=10, xlim=(vmin - buffer, vmax + buffer), ylim=(vmin - buffer, vmax + buffer))        
    # g3.set_axis_labels(xlabel='Target ' + unit , ylabel='Prediction ' + unit)
    
    # g3.savefig('{}/task_{}_kde_e{}.eps'.format(save_path, title), format='eps')

def draw_confusion_matrix(cm, class_label, title, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    cax = ax.matshow(cm, vmin=0, vmax=100)

    fig.colorbar(cax,fraction=0.050, pad=0.06)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticklabels([''] + class_label, fontsize=24, rotation=45)
    ax.set_yticklabels([''] + class_label, fontsize=24)
    plt.tight_layout()
    # fig.suptitle(title)
    fig.savefig('{}conf_matrix_{}.png'.format(save_path, title), format='png')
    plt.close()
    
## With the scale color bar
def draw(arr, title, save_path = "./out/", cmap="viridis", vmin = None, vmax = None, mask = None, isCustomCmap = False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    ax.grid(False)
    # ax.set_title(title)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    if mask is not None:
        im = ax.imshow(mask, alpha=1)
    
    if isCustomCmap is True:
        # clist = [(0,"gray"), (1./100.,"blue"), (495./1000., "white"),  (1, "red")]
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("name", clist)
        oldcmap = matplotlib.cm.get_cmap(cmap, 256)
        newcolors = oldcmap(np.linspace(0, 1, 256))
        black = np.array([0, 0, 0, 1])
        newcolors[:1, :] = black
        cmap = ListedColormap(newcolors)
    
    if (vmin != None):
        im = ax.imshow(arr, cmap=cmap, aspect='auto', interpolation='none', vmin = vmin, vmax = vmax, alpha=1)
    else:
        im = ax.imshow(arr, cmap=cmap, aspect='auto', interpolation='none', alpha=1)
    fig.colorbar(im)
    plt.tight_layout()
    fig.savefig(save_path + title)
    plt.close(fig)

def draw2(arr, title, save_path = "./out/", cmap="viridis", vmin = None, vmax = None, isCustomCmap = False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(title, vmin, vmax)
    if isCustomCmap is True:
        # clist = [(0,"gray"), (1./100.,"blue"), (495./1000., "white"),  (1, "red")]
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("name", clist)
        oldcmap = matplotlib.cm.get_cmap(cmap, 256)
        newcolors = oldcmap(np.linspace(0, 1, 256))
        black = np.array([0, 0, 0, 1])
        newcolors[:1, :] = black
        cmap = ListedColormap(newcolors)
    
    plt.imsave(save_path + title, arr, cmap=cmap, vmin=vmin, vmax=vmax)

def drawHistogram(arr, title, save_path = "./out/", bins = 200):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.savefig(save_path + title)
    plt.close()

def drawRegressionFullImage():
    save_dir = './FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2'
    data_dir = '../data/TAIGA'
    metadata = torch.load('{}/metadata.pt'.format(data_dir))
    regression_tasks = metadata['reg_label_names']
    metadata = torch.load('{}/metadata.pt'.format(data_dir))

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, 15)]
    upper_bound = [35.51, 30.89, 6240, 24.16, 100, 84, 58, 180, 9.66, 6.45, 1, 1, 1]
    for i in range(regression_tasks.shape[0]): # regression_tasks.shape[0]
        print(i)
        label_name = regression_tasks[i]
        # pred
        pred = np.load('{}/pred_reg_{}.npy'.format(save_dir, i))
        # target
        target = np.load('{}/target_reg_{}.npy'.format(save_dir, i))
        # errors
        abs_error = abs(target - pred)
        signed_error = pred - target

        # std
        std = np.load('{}/std_reg_{}.npy'.format(save_dir, i))
        std[target == 0] = np.min(std) - ((np.max(std) - np.min(std)) / 255)
        draw(std, '{}_{}_3_std_3x3.png'.format(i, label_name), "{}/out/".format(save_dir), "viridis", np.min(std), np.max(std), None, True)

        std5x5 = np.load('{}/std5x5_reg_{}.npy'.format(save_dir, i))
        std5x5[target == 0] = np.min(std5x5) - ((np.max(std5x5) - np.min(std5x5)) / 255)
        draw(std5x5, '{}_{}_3_std_5x5.png'.format(i, label_name), "{}/out/".format(save_dir), "viridis", np.min(std5x5), np.max(std5x5), None, True)

        
        vmax = np.max(abs_error.flatten())
        
        print(vmax)
 
        abs_error[target == 0] = -vmax - (vmax * 2 / 255)
        draw(abs_error, '{}_{}_4_absolute_error.png'.format(i, label_name), "{}/out/".format(save_dir), "RdBu", np.min(abs_error), vmax, None, True)

        signed_error[target == 0] = -vmax - (vmax * 2 / 255)
        draw(signed_error, '{}_{}_5_signed_error.png'.format(i, label_name), "{}/out/".format(save_dir), "RdBu", np.min(signed_error), vmax, None, True)
        pred[pred <= 0] = 0
        pred = pred * upper_bound[i]
        target = target * upper_bound[i]
        vmin = min(np.min(pred.flatten()), np.min(target.flatten()))
        vmax = max(np.max(pred.flatten()), np.max(target.flatten()))
        # print(vmin, vmax)

        # pred[target == 0] = vmin - (vmax -vmin / 255)
        # target[target == 0] = vmin - (vmax -vmin / 255)

        # draw(pred, '{}_{}_1_pred.png'.format(i, label_name), "{}/out/".format(save_dir), "viridis", 0, vmax, False)
        # draw(target, '{}_{}_2_target.png'.format(i, label_name), "{}/out/".format(save_dir), "viridis", 0, vmax, False)

        
        draw(pred, '{}_{}_1_pred.png'.format(i, label_name), "{}/out/".format(save_dir), "viridis", vmin, vmax, None, True)

        
        draw(target, '{}_{}_2_target.png'.format(i, label_name), "{}/out/".format(save_dir), "viridis", vmin, vmax, None, True)   

def drawCategoricalFullImage():
    save_dir = './FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2'
    data_dir = '../data/TAIGA'
    metadata = torch.load('{}/metadata.pt'.format(data_dir))
    categorical_tasks = metadata['cls_label_names']
    metadata = torch.load('{}/metadata.pt'.format(data_dir))

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, 15)]

    for i in range(categorical_tasks.shape[0]):
        print(i)
        label_name = categorical_tasks[i]
        # pred
        pred = np.load('{}/pred_reg_{}.npy'.format(save_dir, i))
        # target
        target = np.load('{}/target_reg_{}.npy'.format(save_dir, i))
        # errors
        abs_error = abs(target - pred)
        signed_error = pred - target

        # std
        std = np.load('{}/std_reg_{}.npy'.format(save_dir, i))
        std[target == 0] = np.min(std) - ((np.max(std) - np.min(std)) / 255)
        draw2(std, '{}_{}_3_std_3x3.png'.format(i, label_name), "{}/out/".format(save_dir), "viridis", np.min(std), np.max(std), True)

        
        vmax = np.max(abs_error.flatten())
        
        print(vmax)

        abs_error[target == 0] = -vmax - (vmax * 2 / 255)
        draw2(abs_error, '{}_{}_4_absolute_error.png'.format(i, label_name), "{}/out/".format(save_dir), "RdBu", np.min(abs_error), vmax, True)

        signed_error[target == 0] = -vmax - (vmax * 2 / 255)
        draw2(signed_error, '{}_{}_5_signed_error.png'.format(i, label_name), "{}/out/".format(save_dir), "RdBu", np.min(signed_error), vmax, True)

        vmin = min(np.min(pred.flatten()), np.min(target.flatten()))
        vmax = max(np.max(pred.flatten()), np.max(target.flatten()))
        print(vmin, vmax)

        pred[target == 0] = vmin - (vmax -vmin / 255)
        target[target == 0] = vmin - (vmax -vmin / 255)

        draw2(pred, '{}_{}_1_pred.png'.format(i, label_name), "{}/out/".format(save_dir), "viridis", 0, vmax, False)
        draw2(target, '{}_{}_2_target.png'.format(i, label_name), "{}/out/".format(save_dir), "viridis", 0, vmax, False)

def prepareStandPred():    
## For complete image
    row = 12143
    col = 12826

    test_set = np.load('./complete_image.npy', allow_pickle=True) # coord of pixel-level test set

    # prediction 
    complete_image1 = torch.load('./FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2/complete_image1.pt')
    complete_image2 = torch.load('./FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2/complete_image2.pt')
    complete_image3 = torch.load('./FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2/complete_image3.pt')

    all_target_cls = torch.cat((complete_image1['all_target_cls'], complete_image2['all_target_cls'], complete_image3['all_target_cls']), 0)
    all_pred_cls = torch.cat((complete_image1['all_pred_cls'], complete_image2['all_pred_cls'], complete_image3['all_pred_cls']), 0)
    all_target_reg = torch.cat((complete_image1['all_target_reg'], complete_image2['all_target_reg'], complete_image3['all_target_reg']), 0)
    all_pred_reg = torch.cat((complete_image1['all_pred_reg'], complete_image2['all_pred_reg'], complete_image3['all_pred_reg']), 0)

    full_stand_id = np.load('./stand_ids.npy', allow_pickle=True) 
    stand_id_list = [] # stand id list of test set
    for i in range(test_set.shape[0]):
        stand_id_list.append(full_stand_id[test_set[i][0], test_set[i][1]][0])

    unique_stand_id_list = np.unique(stand_id_list)
    #### CATEGORICAL EVALUATION ####
    categorical_data = []
    categorical_columns = ['standid', 'fertility_class', 'soil_class', 'main_tree_species_class', 'pred_fertility_class', 'pred_soil_class', 'pred_main_tree_species_class']
    all_pred_cls2 = torch.zeros((all_pred_cls.shape[0], 3), dtype=torch.long)
    all_target_cls2 = torch.zeros((all_target_cls.shape[0], 3), dtype=torch.long)
    for i in range(len(unique_stand_id_list)):
        print(i)
        stand_id = unique_stand_id_list[i].astype(np.int32) # get stand_id
        ids = np.where(stand_id_list == unique_stand_id_list[i]) 
        pred_cls = torch.sum(all_pred_cls[ids], 0)
        target_cls = torch.sum(all_target_cls[ids], 0)
        row = [stand_id, target_cls[0:4].argmax(-1).item(), target_cls[4:6].argmax(-1).item(), target_cls[6:9].argmax(-1).item(), pred_cls[0:4].argmax(-1).item(), pred_cls[4:6].argmax(-1).item(), pred_cls[6:9].argmax(-1).item()]
        categorical_data.append(row)
        all_pred_cls2[ids] = torch.tensor([pred_cls[0:4].argmax(-1).item(), pred_cls[4:6].argmax(-1).item(), pred_cls[6:9].argmax(-1).item()])
        all_target_cls2[ids] = torch.tensor([target_cls[0:4].argmax(-1).item(), target_cls[4:6].argmax(-1).item(), target_cls[6:9].argmax(-1).item()])

    categorical_df = pd.DataFrame(categorical_data, columns = categorical_columns)
    categorical_df.to_csv('./stand_categorical_df.csv', index=False)
    categorical_df = pd.read_csv('./stand_categorical_df.csv')

    torch.save(all_pred_cls2, './stand_all_pred_cls.pt')
    torch.save(all_target_cls2, './stand_all_target_cls.pt')


    #### CONTINUOUS EVALUATION ####
    all_pred_reg = np.delete(all_pred_reg, [10, 11, 12], 1)
    all_target_reg = np.delete(all_target_reg, [10, 11, 12], 1)
    all_std_reg = torch.zeros(all_pred_reg.shape)
    columns = ['standid', 'basal_area', 'mean_dbh', 'stem_density', 'mean_height', 'percentage_of_pine', 'percentage_of_spruce', 'percentage_of_birch', 'woody_biomass', 'leaf_area_index', 'effective_leaf_area_index', 'pred_basal_area', 'pred_mean_dbh', 'pred_stem_density', 'pred_mean_height','pred_percentage_of_pine', 'pred_percentage_of_spruce', 'pred_percentage_of_birch', 'pred_woody_biomass', 'pred_leaf_area_index', 'pred_effective_leaf_area_index', 'std_basal_area', 'std_mean_dbh', 'std_stem_density', 'std_mean_height', 'std_percentage_of_pine', 'std_percentage_of_spruce', 'std_percentage_of_birch', 'std_woody_biomass', 'std_leaf_area_index', 'std_effective_leaf_area_index']
    upper_bound = [35.51, 30.89, 6240, 24.16, 100, 84, 58, 180, 9.66, 6.45]
    continuous_data = []
    for i in range(len(unique_stand_id_list)):
        print(i)
        stand_id = unique_stand_id_list[i].astype(np.int32)
        ids = np.where(stand_id_list == unique_stand_id_list[i])
        pred_reg = torch.mean(all_pred_reg[ids], 0) * torch.Tensor(upper_bound)
        target_reg = torch.mean(all_target_reg[ids], 0) * torch.Tensor(upper_bound)
        std_reg = torch.std(all_pred_reg[ids], 0) * torch.Tensor(upper_bound)
        row = [stand_id] + target_reg.tolist() + pred_reg.tolist() + std_reg.tolist()
        continuous_data.append(row)
        all_std_reg[ids] = std_reg
        all_pred_reg[ids] = pred_reg
        all_target_reg[ids] = target_reg
        

    continuous_df = pd.DataFrame(continuous_data, columns = columns)
    continuous_df.to_csv('./stand_continuous_df.csv', index=False)
    # continuous_df = pd.read_csv('./stand_continuous_df.csv')

    # torch.save(all_pred_reg, './stand_all_pred_reg.pt')
    # torch.save(all_target_reg, './stand_all_target_reg.pt')
    torch.save(all_std_reg, './stand_all_std_reg.pt')

    continuous_tasks = all_pred_reg.shape[1]
    pred = np.zeros((row, col, continuous_tasks))
    target = np.zeros((row, col, continuous_tasks))


    # Cannot run on local machine
    # std = np.zeros((row, col, all_std_reg.shape[1]))
    # for i in range(all_std_reg.shape[0]):
    #     print(i)
    #     r, c, _, _, _ = test_set[i]
    #     std_test[r, c] = all_std_reg[i]
    #     if i % 100000 == 0:
    #         print(i)
    

def drawStandLevelRegressionFullImage():
    save_dir = './FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2'
    data_dir = '../data/TAIGA'
    metadata = torch.load('{}/metadata.pt'.format(data_dir))
    regression_tasks = metadata['reg_label_names']
    metadata = torch.load('{}/metadata.pt'.format(data_dir))

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, 15)]
    upper_bound = [35.51, 30.89, 6240, 24.16, 100, 84, 58, 180, 9.66, 6.45, 1, 1, 1]
    for i in range(regression_tasks.shape[0]): # regression_tasks.shape[0]
        print(i)
        label_name = regression_tasks[i]
        pixel_pred = np.load('{}/pred_reg_{}.npy'.format(save_dir, i)) * upper_bound[i]
        # pred
        pred = np.load('{}/stand_pred_reg_{}.npy'.format(save_dir, i))
        # target
        target = np.load('{}/stand_target_reg_{}.npy'.format(save_dir, i))
        # std
        std = np.load('{}/stand_std_reg_{}.npy'.format(save_dir, i))
        # errors
        abs_error = abs(target - pred)
        signed_error = pred - target
        
        vmax = np.max(abs_error.flatten())
        
        # print(vmax)

        abs_error[target == 0] = -vmax - (vmax * 2 / 255)
        draw(abs_error, '{}_{}_stand_4_absolute_error.png'.format(i, label_name), "{}/out3/".format(save_dir), "RdBu", np.min(abs_error), vmax, None, True)

        signed_error[target == 0] = -vmax - (vmax * 2 / 255)
        draw(signed_error, '{}_{}_stand_5_signed_error.png'.format(i, label_name), "{}/out3/".format(save_dir), "RdBu", np.min(signed_error), vmax, None, True)

        vmin = min(np.min(pred.flatten()), np.min(target.flatten()))
        vmax = max(np.max(pred.flatten()), np.max(target.flatten()))
        
        draw(pred, '{}_{}_stand_1_pred.png'.format(i, label_name), "{}/out3/".format(save_dir), "viridis", vmin, vmax, None, True)        
        draw(target, '{}_{}_stand_2_target.png'.format(i, label_name), "{}/out3/".format(save_dir), "viridis", vmin, vmax, None, True)   
        draw(pixel_pred, '{}_{}_pixel_1_pred.png'.format(i, label_name), "{}/out3/".format(save_dir), "viridis", vmin, vmax, None, True)   
        vmin = min(std.flatten())
        vmax = max(std.flatten())

        draw(std, '{}_{}_stand_3_std.png'.format(i, label_name), "{}/out3/".format(save_dir), "viridis", vmin, vmax, None, True)   

def drawStandLevelCategoricalFullImage():
    save_dir = './FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2'
    data_dir = '../data/TAIGA'
    metadata = torch.load('{}/metadata.pt'.format(data_dir))
    categorical_tasks = metadata['cls_label_names']
    metadata = torch.load('{}/metadata.pt'.format(data_dir))

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, 15)]
    for i in range(categorical_tasks.shape[0]): # categorical_tasks.shape[0]
        print(i)
        label_name = categorical_tasks[i]
        # pred
        pred = np.load('{}/stand_pred_cls_{}.npy'.format(save_dir, i))
        # target
        target = np.load('{}/stand_target_cls_{}.npy'.format(save_dir, i))
        # errors
        abs_error = abs(target - pred)
        signed_error = pred - target
        
        vmax = np.max(abs_error.flatten())
        
        # print(vmax)

        abs_error[target == 0] = -vmax - (vmax * 2 / 255)
        draw(abs_error, 'cls_{}_{}_4_absolute_error.png'.format(i, label_name), "{}/out3/".format(save_dir), "RdBu", np.min(abs_error), vmax, None, True)

        signed_error[target == 0] = -vmax - (vmax * 2 / 255)
        draw(signed_error, 'cls_{}_{}_5_signed_error.png'.format(i, label_name), "{}/out3/".format(save_dir), "RdBu", np.min(signed_error), vmax, None, True)

        vmin = min(np.min(pred.flatten()), np.min(target.flatten()))
        vmax = max(np.max(pred.flatten()), np.max(target.flatten()))
        
        draw(pred, 'cls_{}_{}_1_pred.png'.format(i, label_name), "{}/out3/".format(save_dir), "viridis", vmin, vmax)        
        draw(target, 'cls_{}_{}_2_target.png'.format(i, label_name), "{}/out3/".format(save_dir), "viridis", vmin, vmax)   

def drawStandLevelRegressionScatterPlot():
    save_dir = './FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2'
    data_dir = '../data/TAIGA'
    metadata = torch.load('{}/metadata.pt'.format(data_dir))
    regression_tasks = metadata['reg_label_names']
    metadata = torch.load('{}/metadata.pt'.format(data_dir))
    sns.set(font_scale=2)
    sns.set_style("whitegrid", {'axes.grid' : False})

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, 15)]
    # upper_bound = [35.51, 30.89, 6240, 24.16, 100, 84, 58, 180, 9.66, 6.45, 1, 1, 1]
    columns = ['basal_area', 'mean_dbh', 'stem_density', 'mean_height', 'percentage_of_pine', 'percentage_of_spruce', 'percentage_of_birch', 'woody_biomass', 'leaf_area_index', 'effective_leaf_area_index']

    stand_continuous_df = pd.read_csv('./stand_continuous_df.csv')

    for i in range(len(columns)): # regression_tasks.shape[0]
        label_name = columns[i]
        print(label_name)
        # pred
        pred = stand_continuous_df["pred_" + label_name]
        # target
        target = stand_continuous_df[label_name]
        # std
        std = stand_continuous_df["std_" + label_name]
        # errors
        abs_error = abs(target - pred)
        signed_error = pred - target

        draw_scatter_plot(target, pred, "Target", "Prediction", "Target x Prediction - {}_{}".format(i, label_name), "{}/StandLevelRegressionScatterPlot/{}_{}/".format(save_dir, i, label_name), colors[i], True)
        draw_scatter_plot(target, signed_error, "Target", "Signed Error", "Target x Signed Error - {}_{}".format(i, label_name), "{}/StandLevelRegressionScatterPlot/{}_{}/".format(save_dir, i, label_name), colors[i])
        # draw_scatter_plot(target, abs_error, "Target", "Absolute Error", "Target x Absolute Error - {}_{}".format(i, label_name), "{}/StandLevelRegressionScatterPlot/{}_{}/".format(save_dir, i, label_name), colors[i])
        draw_scatter_plot(pred, signed_error, "Prediction", "Signed Error", "Prediction x Signed Error - {}_{}".format(i, label_name), "{}/StandLevelRegressionScatterPlot/{}_{}/".format(save_dir, i, label_name), colors[i])
        # draw_scatter_plot(pred, abs_error, "Prediction", "Absolute Error", "Prediction x Absolute Error - {}_{}".format(i, label_name), "{}/StandLevelRegressionScatterPlot/{}_{}/".format(save_dir, i, label_name), colors[i])

        draw_scatter_plot(target, std, "Target", "Standard Deviation", "Target x Standard Deviation - {}_{}".format(i, label_name), "{}/StandLevelRegressionScatterPlot/{}_{}/".format(save_dir, i, label_name), colors[i])
        draw_scatter_plot(pred, std, "Prediction", "Standard Deviation", "Prediction x Standard Deviation - {}_{}".format(i, label_name), "{}/StandLevelRegressionScatterPlot/{}_{}/".format(save_dir, i, label_name), colors[i])
        draw_scatter_plot(signed_error, std, "Signed Error", "Standard Deviation", "Signed Error x Standard Deviation - {}_{}".format(i, label_name), "{}/StandLevelRegressionScatterPlot/{}_{}/".format(save_dir, i, label_name), colors[i])
        # draw_scatter_plot(abs_error, std, "Absolute Error", "Standard Deviation", "Absolute Error x Standard Deviation - {}_{}".format(i, label_name), "{}/StandLevelRegressionScatterPlot/{}_{}/".format(save_dir, i, label_name), colors[i])


def drawStandLevelConfusionMatrix():
    save_dir = './FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2'
    categorical_df = pd.read_csv('./stand_categorical_df.csv')
    categorical_tasks = ['fertility_class', 'soil_class', 'main_tree_species_class']
    class_labels = [
        ['h-rich', 'mesic', 'sub-x', 'xeric'], 
        ['miner.', 'org.'],
        ['pine', 'spruce', 'birch']
    ]

    for i in range(len(categorical_tasks)):
        label_name = categorical_tasks[i]
        cm = confusion_matrix(categorical_df[label_name].values, categorical_df['pred_' + label_name].values)

        cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.around(100 * cm, decimals=2)
        draw_confusion_matrix(cm, class_labels[i], label_name, "{}/StandLevelConfusionMatrix/".format(save_dir))

def drawPixelLevelRegressionScatterPlot():
# test_set = np.load('./complete_image.npy', allow_pickle=True) # coord of pixel-level test set
    save_dir = './FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2'
    data_dir = '../data/TAIGA'
    metadata = torch.load('{}/metadata.pt'.format(data_dir))
    regression_tasks = metadata['reg_label_names']
    # prediction 
    complete_image1 = torch.load('./FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2/complete_image1.pt')
    complete_image2 = torch.load('./FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2/complete_image2.pt')
    complete_image3 = torch.load('./FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2/complete_image3.pt')

    all_target_reg = torch.cat((complete_image1['all_target_reg'], complete_image2['all_target_reg'], complete_image3['all_target_reg']), 0)
    all_pred_reg = torch.cat((complete_image1['all_pred_reg'], complete_image2['all_pred_reg'], complete_image3['all_pred_reg']), 0)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, 15)]
    for i in range(len(regression_tasks)):
        target = np.load('./FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2/target_reg_{}.npy'.format(i))
        pred = np.load('./FL_Pham34-12012021-gn-aug-eelis-patchsize45_set2/pred_reg_{}.npy'.format(i))
        target = target.flatten()
        pred = pred.flatten()
        target_not_0_indexes = np.where(target != 0)[0]
        pred = pred[[target_not_0_indexes]]
        target = target[[target_not_0_indexes]]

        draw_scatter_plot(target, pred, "Target", "Prediction", "Target x Prediction - {}_{}".format(i, i), "{}/PixelLevelRegressionScatterPlot/{}_{}/".format(save_dir, i, i), colors[i])

def main():
    # drawRegressionFullImage()
    # drawCategoricalFullImage()
    # drawStandLevelRegressionFullImage()
    # drawStandLevelCategoricalFullImage()
    drawStandLevelRegressionScatterPlot()
    # drawStandLevelConfusionMatrix()
    # drawPixelLevelRegressionScatterPlot()

if __name__ == "__main__":
    main()