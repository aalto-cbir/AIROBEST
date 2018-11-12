#! /usr/bin/env python3

# module load python-env/3.5.3
# module load python-env/intelpython3.6-2018.3

import matplotlib

matplotlib.use('Agg')

import numpy as np
import spectral
import matplotlib.pyplot as plt
import sys
import os
import copy
import argparse
from sklearn import decomposition, preprocessing, utils
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from tools.hypdatatools_img import get_geotrans

print(' '.join(sys.argv))

# print('python',     sys.version)
# print('matplotlib', matplotlib.__version__)
# print('sklearn',    sklearn.__version__, flush=True)

dirlist = ['/proj/deepsat/hyperspectral',
           '~/airobest/hyperspectral',
           '.']

for d in dirlist:
    hdir = os.path.expanduser(d)
    if os.path.isdir(hdir):
        break

variable = 'fertilityclass'
threshold = 0
dim = 110
norm = 'l2'
nsamples = 1000
titta_d = -1

parser = argparse.ArgumentParser('PCA analysis of hyperspectral data')
parser.add_argument('--variable', help='Set forest parameter to plot')
parser.add_argument('--norm', help='Set normalization l1|l2|raw')
parser.add_argument('--threshold', help='Set primary specie percentage threshold')
parser.add_argument('--titta_d', help='Set max distance to titta points')
args = parser.parse_args()

if args.variable:
    variable = args.variable

if args.norm:
    norm = args.norm

if args.threshold:
    threshold = args.threshold

if args.titta_d:
    titta_d = int(args.titta_d)

vv = []


def envi2world_p(xy, GT):
    P = xy[0] + 0.5  # relative to pixel corner
    L = xy[1] + 0.5
    xy = (GT[0] + P * GT[1] + L * GT[2],
          GT[3] + P * GT[4] + L * GT[5])
    return xy


def world2envi_p(xy, GT):
    X = xy[0]
    Y = xy[1]
    D = GT[1] * GT[5] - GT[2] * GT[4]
    xy = ((X * GT[5] - GT[0] * GT[5] + GT[2] * GT[3] - Y * GT[2]) / D - 0.5,
          (Y * GT[1] - GT[1] * GT[3] + GT[0] * GT[4] - X * GT[4]) / D - 0.5)
    return xy


def hyp2for(xy0, hgt, fgt):
    xy1 = envi2world_p(xy0, hgt)
    xy2 = world2envi_p(xy1, fgt)
    return (int(np.floor(xy2[0] + 0.1)), int(np.floor(xy2[1] + 0.1)))


pca_model = None


def do_pca(data, n, txt, keep):
    global pca_model

    pca = decomposition.PCA(n_components=n)
    pca.fit(data)
    if keep:
        pca_model = pca

    print(txt, data.shape, pca.components_.shape)

    c = range(len(pca.mean_))

    f = plt.figure('pca-' + txt)
    plt.title(txt)
    p1 = plt.subplot(311)
    p1.plot(c, pca.mean_)

    cc = []
    for i in pca.components_:
        if np.dot(pca.mean_, i) < 0:
            cc.append(-i)
        else:
            cc.append(i)

    p2 = plt.subplot(312)
    z = np.zeros(len(c))
    p2.plot(c, z, 'k:', c, cc[0], 'b', c, cc[1], 'g', c, cc[2], 'r')

    p3 = plt.subplot(313)

    val = pca.explained_variance_ / pca.explained_variance_[0]
    idx = range(len(val))

    # print(val)
    p3.plot(idx, val, 'bo', idx, val, 'k')
    p3.set_yscale('log')

    # f.show()
    f.savefig('pca-' + txt + '.pdf')

    # vvv = [pca.mean_]
    vvv = np.concatenate(([pca.mean_], cc[0:1]))
    # vvv = np.concatenate(([pca.mean_], cc[0:3]))
    for v in vvv:
        vv.append(v / np.linalg.norm(v))


hyphdr = hdir + '/subset_A_20170615_reflectance.hdr'
# hyphdr  = '../20170615_reflectance_mosaic_128b.hdr'
hypdata = spectral.open_image(hyphdr)
hypmap = hypdata.open_memmap()
hypgt = get_geotrans(hyphdr)
print('Spectral: {:d} x {:d} pixels, {:d} bands in {:s}'. \
      format(hypdata.shape[1], hypdata.shape[0], hypdata.shape[2], hyphdr))

forhdr = hdir + '/forestdata.hdr'
fordata = spectral.open_image(forhdr)
formap = fordata.open_memmap()
forgt = get_geotrans(forhdr)
print('Forest:   {:d} x {:d} pixels, {:d} bands in {:s}'. \
      format(fordata.shape[1], fordata.shape[0], fordata.shape[2], forhdr),
      flush=True)

titta_params = []
titta_world = []
titta_xy = []
titta_val = []
tittaname = (hdir + '/Titta2013_coords.txt', 'utf-8')
tittaname = (hdir + '/Titta2013.txt', 'utf-16')
with open(tittaname[0], encoding=tittaname[1]) as tf:
    ln = 0
    for ll in tf:
        lx = ll.rstrip().split('\t')
        # print(lx)
        if ln == 0:
            titta_params = lx[3:]
        else:
            idxy = [int(lx[0]), int(lx[1]), int(lx[2])]
            titta_world.append(idxy)
            xy = world2envi_p(idxy[1:3], hypgt)
            titta_xy.append(xy)
            v = []
            for i in range(3, len(lx)):
                vf = 0.0;
                if lx[i] != '#DIV/0!':
                    vf = float(lx[i])
                v.append(vf)
            titta_val.append(v)
        ln += 1
print('Titta:   {:d} points, {:d} variables in {:s}'. \
      format(len(titta_val), len(titta_val[0]), tittaname[0]),
      flush=True)

if False:
    hxy = (0, 0)
    fxy = hyp2for(hxy, hypgt, forgt)
    print(hxy, fxy)
    hxy = (1, 1)
    fxy = hyp2for(hxy, hypgt, forgt)
    print(hxy, fxy)
    hxy = (hypdata.shape[1] - 1, hypdata.shape[0] - 1)
    fxy = hyp2for(hxy, hypgt, forgt)
    print(hxy, fxy)

forest_params = fordata.metadata['band names']
for f in range(len(forest_params)):
    p = forest_params[f].find('_[')
    if p >= 0:
        forest_params[f] = forest_params[f][:p]
    p = forest_params[f].find('*')
    if p >= 0:
        forest_params[f] = forest_params[f][:p]

if titta_d != -1:
    forest_params = titta_params

print('Variables:', ' '.join(forest_params))
variable_idx = {}
idx = 0
for v in forest_params:
    variable_idx[v] = idx
    idx += 1
vidx = variable_idx[variable]
percentage_mainspecies_idx = variable_idx['percentage_mainspecies']
print('Variable', variable, 'maps to component index', vidx)
print('percentage_mainspecies maps to component index', percentage_mainspecies_idx)

print('Normalization', norm)
print('Threshold', threshold)
print('Titta_d', titta_d, flush=True)

wavelength_list = hypdata.metadata['wavelength']
wavelength = np.array(wavelength_list, dtype=float)
i_r = abs(wavelength - 0.660).argmin()  # red band, the closest to 660 nm
i_g = abs(wavelength - 0.550).argmin()  # green, closest band to 550 nm
i_b = abs(wavelength - 0.490).argmin()  # blue, closest to 490 nm

hyp_rgb = np.zeros((hypdata.shape[0], hypdata.shape[1], 3), np.float32)
cls_rgb = copy.copy(hyp_rgb)

ydata = []
data = {}
prodata = {}
if True:
    for r in range(hypdata.shape[0]):
        # print('a', r)
        for c in range(hypdata.shape[1]):
            h = hypmap[r, c]
            # rgb = (np.sum(h[80:128]), np.sum(h[40:80]), np.sum(h[0:40]))
            rgb = np.array((h[i_r], h[i_g], h[i_b])) / 8
            # print(rgb)
            hyp_rgb[r, c] = rgb

            tittahit = -1
            if True and titta_d >= 0:
                for i in range(len(titta_xy)):
                    xy = titta_xy[i]
                    if xy[1] >= r - titta_d - 0.5 and xy[1] <= r + titta_d + 0.5 and \
                            xy[0] >= c - titta_d - 0.5 and xy[0] <= c + titta_d + 0.5:
                        tittahit = i
                        print('TITTA', r, c, i, flush=True)
                        break
            intitta = tittahit != -1

            if intitta:
                v = titta_val[tittahit]
            else:
                xy = hyp2for((c, r), hypgt, forgt)
                v = formap[xy[1], xy[0]]

            ydata.append(v)
            x = v[percentage_mainspecies_idx]  # percentage_mainspecies
            if variable != 'maintreespecies':
                x = 100
            s = v[vidx]
            # print(r, c, s, x)

            if (s == 0):
                rgb = (255, 0, 0)
            elif (s == 1):
                rgb = (0, 255, 0)
            elif (s == 2):
                rgb = (0, 0, 255)
            elif (s == 3 or s == 29):
                rgb = (0, 255, 255)
            elif (s == 4):
                rgb = (255, 0, 255)
            elif (s == 5):
                rgb = (255, 255, 0)
            elif (s >= 6 and s <= 11):
                rgb = (0, 0, 0)
            else:
                rgb = (128, 128, 128)

            if titta_d == -1 and x >= threshold or intitta:
                if not s in data:
                    data[s] = []
                data[s].append(h)
            elif titta_d == -1:
                rgb = (196, 196, 196)

            if True:
                if intitta:
                    rgb = (255, 255, 255)

            cls_rgb[r, c] = rgb

            if intitta:
                if not -2 in data:
                    data[-2] = []
                data[-2].append(h)

            if not -1 in data:
                data[-1] = []
            data[-1].append(h)

if False:
    dd = 5
    for xy in titta:
        xy2 = world2envi_p(xy[1:3], hypgt)
        xx = int(np.floor(xy2[0] + 0.1))
        yy = int(np.floor(xy2[1] + 0.1))
        if xx >= 0 and xx < hypdata.shape[1] and \
                yy >= 0 and yy < hypdata.shape[0]:
            print('HIT', xy[0], xx, yy)
            for x in range(xx - dd, xx + dd + 1):
                for y in range(yy - dd, yy + dd + 1):
                    if x >= 0 and x < hypdata.shape[1] and \
                            y >= 0 and y < hypdata.shape[0]:
                        cls_rgb[y, x] = (255, 255, 255)

tix = ''
if titta_d >= 0:
    tix = '-titta' + str(titta_d)

Image.fromarray(hyp_rgb.astype('uint8')).save('hyp_rgb.png')
Image.fromarray(cls_rgb.astype('uint8')).save(variable + tix + '.png')

# data[-1] = np.reshape(hypmap, (-1, hypdata.shape[2]))

if True:
    for h in data.keys():
        d = np.array(data[h])
        d = d[:, :dim]
        if norm == 'l2' or norm == 'l1':
            d = preprocessing.normalize(d, norm=norm)
        prodata[h] = d
        # print(h, d.shape)
        # vv = []
        hx = str(h)
        if hx == '-1':
            hx = 'full'
        if hx == '-2':
            hx = tix[1:]
        do_pca(d, 20, hx + '-' + variable + '-' + norm + '-' + str(threshold) + tix, h == -1)

if False:
    print('starting to save numpy', flush=True)
    np.save('x-' + str(dim) + '-' + norm + '.npy', prodata[-1])
    np.save('y.npy', np.array(ydata))
    print('done saving numpy', flush=True)

if False:
    for a in vv:
        for b in vv:
            d = np.dot(a, b)
            print(d, end=' ')
            if False:
                if (d > 1e-10):
                    print('{:.3f}'.format(np.log10(d)), end=' ')
                else:
                    print('xxx', end=' ')
        print()

if True:
    cn = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'k', 'k', 'k', 'k', 'k']
    s = (plt.rcParams['lines.markersize'] / 8.0) ** 2
    for xy in ['01', '12']:
        ci = 0
        f = plt.figure('pca mapped data ' + norm + ' ' + xy)
        for h in sorted(prodata.keys()):
            if h >= 0:
                c = cn[ci]
                print(xy, h, ci, c, flush=True)
                zzx = pca_model.transform(prodata[h])
                if xy == '01':
                    zz = zzx[:, :2]
                else:
                    zz = zzx[:, 1:3]
                ns = nsamples
                if zz.shape[0] < ns:
                    ns = zz.shape[0]
                zz = utils.resample(zz, replace=False,
                                    n_samples=ns, random_state=0)
                plt.scatter(zz[:, 0], zz[:, 1], s=s, c=c)
                ci += 1
        # f.show()
        f.savefig('pca-mapped-data-' + variable + '-' + norm + tix + '-' + xy + '.pdf');
