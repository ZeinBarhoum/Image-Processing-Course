import time

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import NoNorm

Num = 65
GlobalLungs1 = [np.zeros((128, 256), np.uint8) for i1 in range(Num)]
GlobalLungs2 = [np.zeros((128, 256), np.uint8) for j1 in range(Num)]
GlobalMLungs1 = [np.zeros((128, 256), np.uint8) for i2 in range(Num)]
GlobalMLungs2 = [np.zeros((128, 256), np.uint8) for j2 in range(Num)]
GlobalMasks1 = [np.zeros((128, 256), np.uint8) for i3 in range(Num)]
GlobalMasks2 = [np.zeros((128, 256), np.uint8) for j3 in range(Num)]
GlobalDis1 = [np.zeros((128, 256), float) for i4 in range(Num)]
GlobalDis2 = [np.zeros((128, 256), float) for j4 in range(Num)]
GlobalAnngles1 = [np.zeros((128, 256), float) for i5 in range(Num)]
GlobalAnngles2 = [np.zeros((128, 256), float) for j5 in range(Num)]

Globalang1 = [0 for i6 in range(Num)]
Globalang2 = [0 for j6 in range(Num)]
GlobalThresholds1 = [[0, 0, 0] for i7 in range(Num)]
GlobalThresholds2 = [[0, 0, 0] for j7 in range(Num)]


def Section(j):
    if (j > 200):
        return 3
    elif (j < 50):
        return 1
    else:
        return 2


def good(I, F):
    if (I < 125):
        if (F > 100):
            return False
        else:
            return True

    return True


def eq(I1, I2, D1, D2, A1, A2, T1, T2):
    e1 = abs(I1 - I2)
    e2 = 5 * abs(D1 - D2)
    if (D1 < 60):
        e3 = 0
    else:
        e3 = abs(A1 - A2)
    e4 = abs(T1[0] + T1[1] + T1[2] - T2[0] - T2[1] - T2[2])
    return e1 + e2 + e3 + e4


def Find(I, D, A, T, x, lung):
    xmin, xmax, ymin, ymax = x
    wanted = [-1, -1, -1]
    min = 10000000000
    min1 = 10000000000
    ks = []
    if (lung == 1):
        for k in range(Num):
            T1 = T
            T2 = GlobalThresholds1[k]
            e = abs(T1[0] + T1[1] + T1[2] - T2[0] - T2[1] - T2[2])
            ks.append((e, k))

        ks = sorted(ks)
        for s in range(3):
            k = ks[s][1]
            for i in range(xmin, xmax):
                for j in range(ymin, ymax):
                    if (GlobalMasks1[k][i][j] != 0 and GlobalLungs1[k][i][j] > I - 7 and GlobalLungs1[k][i][j] < I + 7):
                        e = eq(I, GlobalLungs1[k][i][j], D, GlobalDis1[k][i][j], A, GlobalAnngles1[k][i][j], T,
                               GlobalThresholds1[k])
                        if (e < min and good(I, GlobalMLungs1[k][i][j])):
                            min = e
                            wanted = [k, i, j]

    if (lung == 2):
        for k in range(Num):
            T1 = T
            T2 = GlobalThresholds2[k]
            e = abs(T1[0] + T1[1] + T1[2] - T2[0] - T2[1] - T2[2])
            ks.append((e, k))

        ks = sorted(ks)
        for s in range(3):
            k = ks[s][1]
            for i in range(xmin, xmax):
                for j in range(ymin, ymax):
                    if (GlobalMasks2[k][i][j] != 0 and GlobalLungs2[k][i][j] > I - 7 and GlobalLungs2[k][i][j] < I + 7):
                        e = eq(I, GlobalLungs2[k][i][j], D, GlobalDis2[k][i][j], A, GlobalAnngles2[k][i][j], T,
                               GlobalThresholds2[k])
                        if (e < min and good(I, GlobalMLungs2[k][i][j])):
                            min = e
                            wanted = [k, i, j]
    return wanted


def LoadAll():
    global GlobalLungs1, GlobalLungs2, GlobalMLungs1, GlobalMLungs2, GlobalMasks1, GlobalMasks2, GlobalDis1, GlobalDis2, GlobalAnngles1, GlobalAnngles2, Globalang1, Globalang2, GlobalThresholds1, GlobalThresholds2
    GlobalLungs1 = np.load('GlobalLungs1.npy')
    GlobalLungs2 = np.load('GlobalLungs2.npy')
    GlobalMLungs1 = np.load('GlobalMLungs1.npy')
    GlobalMLungs2 = np.load('GlobalMLungs2.npy')
    GlobalMasks1 = np.load('GlobalMasks1.npy')
    GlobalMasks2 = np.load('GlobalMasks2.npy')
    GlobalDis1 = np.load('GlobalDis1.npy')
    GlobalDis2 = np.load('GlobalDis2.npy')
    GlobalAnngles1 = np.load('GlobalAnngles1.npy')
    GlobalAnngles2 = np.load('GlobalAnngles2.npy')
    Globalang1 = np.load('Globalang1.npy')
    Globalang2 = np.load('Globalang2.npy')
    GlobalThresholds1 = np.load('GlobalThresholds1.npy')
    GlobalThresholds2 = np.load('GlobalThresholds2.npy')


def StoreAll():
    global GlobalDis1
    global GlobalDis2

    for k in range(Num):
        GlobalLungs1[k], GlobalLungs2[k], GlobalMLungs1[k], GlobalMLungs2[k], GlobalMasks1[k], GlobalMasks2[k] = Lungs(
            k + 1)
        m1 = GetMiddle(GlobalMasks1[k])
        m2 = GetMiddle(GlobalMasks2[k])

        for i in range(0, 128):
            for j in range(0, 256):
                GlobalDis1[k][i][j] = dis((i, j), m1)
                GlobalDis2[k][i][j] = dis((i, j), m2)
                GlobalAnngles1[k][i][j] = angel((j - m1[1], m1[0] - i), (0, 0))
                GlobalAnngles2[k][i][j] = angel((j - m2[1], m2[0] - i), (0, 0))
        # GlobalDis1[k] = GlobalDis1[k] / np.max(GlobalDis1[k])
        # GlobalDis2[k] = GlobalDis2[k] / np.max(GlobalDis2[k])
        GlobalThresholds1[k] = GetTripleOtsuFast(GlobalLungs1[k], GlobalMasks1[k])
        GlobalThresholds2[k] = GetTripleOtsuFast(GlobalLungs2[k], GlobalMasks2[k])


def GetDft(img):
    f = np.fft.fft2(img)
    return f


def CenterDft(f):
    f1 = np.copy(f)
    f1 = np.fft.fftshift(f)
    return f1


def normalize(img):
    min = np.min(img)
    max = np.max(img)
    img1 = np.copy(img)
    img1 = np.round((img1 - min) * 255 / (max - min))
    img1 = img1.astype(np.uint8)
    return img1


def GetIdft(f):
    img = np.fft.ifft2(f)
    img = normalize(np.abs(img))
    img = img.astype(np.uint8)
    return img


def GetLoG(size, sigma, Max_wanted=-1):
    f = np.zeros(shape=size, dtype=float)  # Create the filter with zero values
    midx = int(size[0] / 2)  # x of the center
    midy = int(size[1] / 2)  # y of the center
    for x in range(size[0]):  # iterate over the filter
        for y in range(size[1]):
            # Calculate the LoG(x,y)
            x2 = 1.0 * np.power(x - midx, 2)
            y2 = 1.0 * np.power(y - midy, 2)
            s2 = 1.0 * np.power(sigma, 2)
            exp = 1.0 * np.exp(-1.0 * (x2 + y2) / (2 * s2))
            k = 1.0 * (x2 + y2 - 2 * s2) / (np.pi * np.power(sigma, 6))
            f[x][y] = k * exp

    f[midx][midy] -= sum(sum(f))  # the sum of all values should be 0
    f = f * Max_wanted / (f[midx][midy])
    return f


def pad(img, px, py):
    img1 = np.zeros(shape=(px, py), dtype=img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img1[i][j] = img[i][j]
    return img1


def ApplyKernal(img, f):
    px = img.shape[0] + f.shape[0] - 1
    py = img.shape[1] + f.shape[1] - 1
    pimg = pad(img, px, py)
    pfil = pad(f, px, py)
    ftimg = GetDft(pimg)
    ftfil = GetDft(pfil)
    ftnew = np.multiply(ftimg, ftfil)
    pnimg = GetIdft(ftnew)
    nimg = np.zeros(img.shape, dtype=img.dtype)
    for i in range(nimg.shape[0]):
        for j in range(nimg.shape[1]):
            nimg[i][j] = pnimg[i + int((f.shape[0] - 1) / 2)][j + int((f.shape[1] - 1) / 2)]
    return nimg


def ApplyLoG(img, size, sigma):
    f = GetLoG(size, sigma, 1)
    filterdImg = ApplyKernal(img, f)
    float_img = img.astype(float)
    float_filterd = filterdImg.astype(float)
    addition = float_img - float_filterd
    # addition=float_filterd
    addition = normalize(addition)
    return addition


def GetHistogramImage(img, mask, a=0, b=255):
    his = np.zeros(shape=[256, 256], dtype=np.uint8)
    nop = np.zeros(shape=256, dtype=float)
    max = 0
    num = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (mask[i][j] != 0 and img[i][j] >= a and img[i][j] <= b):
                nop[img[i][j]] += 1

    # print(nop)
    max = np.max(nop)
    for i in range(256):
        nop[i] = round(nop[i] * 256 / max)
        for j in range(int(256 - nop[i])):
            his[j][i] = 255
    return his


def GetHistogram(img, mask, a, b):
    nop = np.zeros(shape=256, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (mask[i][j] != 0 and img[i][j] >= a and img[i][j] <= b):
                nop[img[i][j]] += 1

    return nop


def GetOtsu(img, mask, a, b):
    hist = GetHistogram(img, mask, a, b)
    MN = sum(hist)
    p = hist * 1.0 / MN
    p1 = np.copy(p)
    m = np.copy(p)
    bcv = np.copy(p)
    sum1 = 0
    sum2 = 0
    for r in range(0, 256):
        i = r
        sum1 += p[i]
        sum2 += i * p[i]
        p1[i] = sum1
        m[i] = sum2
    mg = m[255]
    max = -1
    k = -1
    for r in range(a, b + 1):
        i = r
        bcv[i] = np.power(mg * p1[i] - m[i], 2) / (p1[i] * (1 - p1[i]))
        if (bcv[i] > max):
            max = bcv[i]
            k = r
    return k


def GetTripleOtsuFast(img, mask, a=0, b=255):
    hist = GetHistogram(img, mask, a, b)
    MN = sum(hist)
    p = hist / MN
    mg = 0.0
    k = [0, 0, 0]
    sumarr1 = np.zeros(256, float)
    sumarr2 = np.zeros((256, 256), float)
    sumarr4 = np.zeros(256, float)
    meanarr1 = np.zeros(256, float)
    meanarr2 = np.zeros((256, 256), float)
    meanarr4 = np.zeros(256, float)
    sumarr1[0] = p[0]
    sumarr4[255] = p[255]
    meanarr4[255] = 255 * p[255]
    for i in range(1, 256):
        mg += i * p[i]
        sumarr1[i] = sumarr1[i - 1] + p[i]
        meanarr1[i] = meanarr1[i - 1] + i * p[i]
        sumarr4[255 - i] = sumarr4[256 - i] + p[255 - i]
        meanarr4[255 - i] = meanarr4[256 - i] + (255 - i) * p[255 - i]
    for i in range(1, 252):
        for j in range(i + 1, 253):
            sumarr2[i][j] = sumarr1[j] - sumarr1[i]
            meanarr2[i][j] = meanarr1[j] - meanarr1[i]
    max = 0
    for k1 in range(1, 90):
        for k2 in range(100, 150):
            for k3 in range(150, 200):
                sum1 = sumarr1[k1]
                mean1 = meanarr1[k1]
                if (sum1 > 0.01):
                    mean1 = mean1 / sum1
                else:
                    sum1 = 0
                sum2 = sumarr2[k1][k2]
                mean2 = meanarr2[k1][k2]
                if (sum2 > 0.01):
                    mean2 = mean2 / sum2
                else:
                    sum2 = 0
                sum4 = sumarr4[k3 + 1]
                mean4 = meanarr4[k3 + 1]
                if (sum4 > 0.01):
                    mean4 = mean4 / sum4
                else:
                    sum4 = 0
                sum3 = 1 - sum1 - sum2 - sum4
                mean3 = mg - sum1 * mean1 - sum2 * mean2 - sum4 * mean4
                if (sum3 > 0.01):
                    mean3 = mean3 / sum3
                else:
                    sum3 = 0
                bcv = sum1 * pow(mean1 - mg, 2) + sum2 * pow(mean2 - mg, 2) + sum3 * pow(mean3 - mg, 2) + sum4 * pow(
                    mean4 - mg, 2)
                if (bcv > max):
                    max = bcv
                    k[0] = k1
                    k[1] = k2
                    k[2] = k3

    return k


def GetImage(k):
    n = np.copy(k)
    name = "covid19/tr_im/tr_im_z"
    a = str(n % 10)
    n = int(n / 10)
    b = str(n % 10)
    n = int(n / 10)
    c = str(n % 10)
    name = name + c + b + a + ".png"
    img = cv.imread(name, 0)
    return img


def GetMask(k):
    n = np.copy(k)
    name = "covid19/tr_lungmasks_updated/tr_lungmasks_updated_z"
    a = str(n % 10)
    n = int(n / 10)
    b = str(n % 10)
    n = int(n / 10)
    c = str(n % 10)
    name = name + c + b + a + ".png"
    img = cv.imread(name, 0)
    return img


def GetManual(k):
    n = np.copy(k)
    name = "covid19/CorrectMasks/Correct_"
    a = str(n % 10)
    n = int(n / 10)
    b = str(n % 10)
    n = int(n / 10)
    c = str(n % 10)
    name = name + c + b + a + ".png"
    img = cv.imread(name, 0)
    return img


def SeparateLungs(img, mask):
    mask1 = mask / 2
    mask1 = mask1.astype(np.uint8)
    mask2 = mask % 2
    mask2 = mask2.astype(np.uint8)
    img1 = img * mask1
    img2 = img * mask2
    Lung1 = CropBoarders(img1, np.where(mask1 > 0))
    Lung2 = CropBoarders(img2, np.where(mask2 > 0))
    return [Lung1, Lung2]


def CropBoarders(img, arr):
    x = arr[0]
    y = arr[1]
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    size = [xmax - xmin + 1, ymax - ymin + 1]
    new = np.zeros(size, np.uint8)
    for i in range(xmin, xmax + 1):
        for j in range(ymin, ymax + 1):
            new[i - xmin][j - ymin] = img[i][j]
    return new


def Lungs(k):
    image = GetImage(k)
    image = cv.GaussianBlur(image, (5, 5), 0)
    manual = GetManual(k)
    mask = GetMask(k)
    lung1, lung2 = SeparateLungs(image, mask)
    mlung1, mlung2 = SeparateLungs(manual, mask)
    mask1, mask2 = SeparateLungs(mask, mask)
    lung1 = cv.resize(lung1, (256, 128))
    lung2 = cv.resize(lung2, (256, 128))
    mlung1 = cv.resize(mlung1, (256, 128))
    mlung2 = cv.resize(mlung2, (256, 128))
    mask1 = cv.resize(mask1, (256, 128))
    mask2 = cv.resize(mask2, (256, 128))

    p1 = [-1, -1]
    p2 = [-1, -1]
    for i in range(128):
        for j in range(20):
            if (mask1[i][j] != 0):
                p1 = [i, j]
                break
        if (p1 != [-1, -1]): break

    for i in range(128):
        for j in range(200, 255):

            if (mask1[i][j] != 0):
                p2 = [i, j]
                break
        if (p2 != [-1, -1]): break
    p1 = [p1[1] - 128, 64 - p1[0]]
    p2 = [p2[1] - 128, 64 - p2[0]]
    an1 = angel(p1, (0, 0))
    an2 = angel(p2, (0, 0))
    an = 90 - (an1 + an2) / 2
    # Globalang1[k - 1] = an
    lung1 = rotate_image(lung1, an, (64, 128))
    mlung1 = rotate_image(mlung1, an, (64, 128))
    mask1 = rotate_image(mask1, an, (64, 128))
    lung1 = CropBoarders(lung1, np.where(mask1 > 0))
    mlung1 = CropBoarders(mlung1, np.where(mask1 > 0))
    mask1 = CropBoarders(mask1, np.where(mask1 > 0))
    lung1 = cv.resize(lung1, (256, 128))
    mlung1 = cv.resize(mlung1, (256, 128))
    mask1 = cv.resize(mask1, (256, 128))

    p1 = [-1, -1]
    p2 = [-1, -1]
    i = lung2.shape[0] - 1
    while (i >= 0):
        for j in range(20):
            if (mask2[i][j] != 0):
                p1 = [i, j]
                break
        if (p1 != [-1, -1]): break
        i -= 1
    i = lung2.shape[0] - 1
    while (i >= 0):
        for j in range(220, 255):
            if (mask2[i][j] != 0):
                p2 = [i, j]
                break
        if (p2 != [-1, -1]): break
        i -= 1
    p1 = [p1[1] - 128, 64 - p1[0]]
    p2 = [p2[1] - 128, 64 - p2[0]]
    an1 = angel(p1, (0, 0))
    an2 = angel(p2, (0, 0))
    an = 90 - (an1 + an2) / 2
    # Globalang2[k - 1] = an
    lung2 = rotate_image(lung2, an, (64, 128))
    mlung2 = rotate_image(mlung2, an, (64, 128))
    mask2 = rotate_image(mask2, an, (64, 128))
    lung2 = CropBoarders(lung2, np.where(mask2 > 0))
    mlung2 = CropBoarders(mlung2, np.where(mask2 > 0))
    mask2 = CropBoarders(mask2, np.where(mask2 > 0))
    lung2 = cv.resize(lung2, (256, 128))
    mlung2 = cv.resize(mlung2, (256, 128))
    mask2 = cv.resize(mask2, (256, 128))

    return [lung1, lung2, mlung1, mlung2, mask1, mask2]


def rotate_image(image, angle, image_center):
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, (1024, 1024), flags=cv.INTER_LINEAR)
    return result


def angel(a, b):
    return 90 - (np.arctan2(a[0] - b[0], a[1] - b[1])) * 180 / np.pi


def RemoveAirFat(lungs, th):
    ret1, tlung1 = cv.threshold(lungs[0], th, 255, cv.THRESH_TOZERO_INV)
    ret2, tlung2 = cv.threshold(lungs[1], th, 255, cv.THRESH_TOZERO_INV)
    return [tlung1, tlung2]


def DualThreshold(image, T):
    ret1, img1 = cv.threshold(image, T[0], 255, cv.THRESH_BINARY)
    ret2, img2 = cv.threshold(image, T[1], 255, cv.THRESH_BINARY)
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    img = (img1 + img2) / 2
    img = img.astype(np.uint8)
    return img


def TripleThreshold(image, T):
    ret1, img1 = cv.threshold(image, T[0], 255, cv.THRESH_BINARY)
    ret2, img2 = cv.threshold(image, T[1], 255, cv.THRESH_BINARY)
    ret3, img3 = cv.threshold(image, T[2], 255, cv.THRESH_BINARY)
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    img3 = img3.astype(float)
    img = (img1 + img2 + img3) / 3
    img = img.astype(np.uint8)
    return img


def Kth_filter(img, size, k):
    x = size[0]
    y = size[1]
    img1 = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            a = []
            for r in range(int(-(x - 1) / 2), int((x - 1) / 2) + 1):
                for c in range(int(-(y - 1) / 2), int((y - 1) / 2) + 1):
                    m = i + r
                    n = j + c
                    if m < 0: m = 0
                    if n < 0: n = 0
                    if m >= img.shape[0]: m = img.shape[0] - 1
                    if n >= img.shape[1]: n = img.shape[1] - 1
                    a.append(img[m][n])
            a.sort()
            img1[i][j] = a[k]
    return img1


def GetMiddle(img):
    xall, yall = np.where(img != 0)
    x = sum(xall) / len(xall)
    y = sum(yall) / len(yall)
    return [x, y]


def Threshold(imgs, k, mode):
    ret, tlung1 = cv.threshold(imgs[0], k, 255, mode)
    ret, tlung2 = cv.threshold(imgs[1], k, 255, mode)
    return (tlung1, tlung2)


def ApplyGaussian(imgs, size):
    glung1 = cv.GaussianBlur(imgs[0], size, 0)
    glung2 = cv.GaussianBlur(imgs[1], size, 0)
    return (glung1, glung2)


def Show(img, n, m, k):
    plt.subplot(n, m, k), plt.imshow(img, cmap='gray', norm=NoNorm())


def ShowAll(imgs):
    m = 2
    n = int(np.ceil(len(imgs) / 2))
    for i in range(n):
        Show(imgs[2 * i], n, m, 2 * i + 1)
        Show(imgs[2 * i + 1], n, m, 2 * i + 2)


def TLungs(Original, Mask):
    m = [GetMiddle(Original[0]), GetMiddle(Original[1])]
    lung1 = np.copy(Original[0])
    lung2 = np.copy(Original[1])
    lungs = [lung1, lung2]
    for z in (0, 1):
        xb1, yb1 = np.where(Original[z] < 60)
        xb = []
        yb = []
        for i in range(len(xb1)):
            if (Mask[z][xb1[i]][yb1[i]] != 0):
                xb.append(xb1[i])
                yb.append(yb1[i])
        xw1, yw1 = np.where(Original[z] > 185)
        xg = []
        yg = []
        xw = []
        yw = []
        distances = []
        for i in range(len(xw1)):
            distances.append(dis([xw1[i], yw1[i]], m[z]))
        maxd = np.max(distances)
        for i in range(len(xw1)):
            if (distances[i] < 0.75 * maxd):
                xb.append(xw1[i])
                yb.append(yw1[i])
            elif (distances[i] < 0.75 * maxd):
                xg.append(xw1[i])
                yg.append(yw1[i])
            else:
                xw.append(xw1[i])
                yw.append(yw1[i])
        for i in range(len(xw)):
            lungs[z][xw[i]][yw[i]] = 255
        for i in range(len(xg)):
            lungs[z][xg[i]][yg[i]] = 200
        for i in range(len(xb)):
            lungs[z][xb[i]][yb[i]] = 0

    return lungs


def SLungs(Original, Mask):
    lung1 = np.copy(Original[0])
    lung2 = np.copy(Original[1])
    o1 = GetOtsu(lung1, Mask[0], 90, 185)
    o2 = GetOtsu(lung2, Mask[1], 90, 185)
    for i in range(lung1.shape[0]):
        for j in range(lung1.shape[1]):
            if (lung1[i][j] >= 90 and lung1[i][j] <= o1):
                lung1[i][j] = 85
            elif (lung1[i][j] > o1 and lung1[i][j] <= 185):
                lung1[i][j] = 170
            elif (lung1[i][j] == 200):
                lung1[i][j] = 170
    for i in range(lung2.shape[0]):
        for j in range(lung2.shape[1]):
            if (lung2[i][j] >= 90 and lung2[i][j] <= o2):
                lung2[i][j] = 85
            elif (lung2[i][j] > o2 and lung2[i][j] <= 185):
                lung2[i][j] = 170
            elif (lung2[i][j] == 200):
                lung2[i][j] = 170
    return [lung1, lung2]


def dis(a, b):
    return np.sqrt(np.power(a[0] - b[0], 2) + np.power(a[1] - b[1], 2))


def Method3(kk):
    LoadAll()
    lung1 = GlobalLungs1[kk - 1]
    lung2 = GlobalLungs2[kk - 1]
    mlung1 = GlobalMLungs1[kk - 1]
    mlung2 = GlobalMLungs2[kk - 1]
    mask1 = GlobalMasks1[kk - 1]
    mask2 = GlobalMasks2[kk - 1]
    m1 = GetMiddle(lung1)
    m2 = GetMiddle(lung2)
    T1 = GlobalThresholds1[kk - 1]
    T2 = GlobalThresholds2[kk - 1]

    for q in range(0, 128):
        for w in range(0, 256):
            if (mask1[q][w] != 0 and lung1[q][w] > 70):
                D = dis((q, w), m1)
                A = angel((w - m1[1], m1[0] - q), (0, 0))
                I = lung1[q][w]
                x = [max(0, q - 3), min(128, q + 3), max(0, w - 3), min(256, w + 3)]
                T = T1
                f = Find(I, D, A, T, x, 1)
                if (f == [-1, -1, -1]):
                    lung1[q][w] = 0
                else:
                    lung1[q][w] = GlobalMLungs1[f[0]][f[1]][f[2]]
            else:
                lung1[q][w] = 0

            if (mask2[q][w] != 0 and lung2[q][w] > 70):
                D = dis((q, w), m2)
                A = angel((w - m2[1], m2[0] - q), (0, 0))
                I = lung2[q][w]
                x = [max(0, q - 3), min(128, q + 3), max(0, w - 3), min(256, w + 3)]
                T = T2
                f = Find(I, D, A, T, x, 2)
                if (f == [-1, -1, -1]):
                    lung2[q][w] = 0
                else:
                    lung2[q][w] = GlobalMLungs2[f[0]][f[1]][f[2]]
            else:
                lung2[q][w] = 0
        print(q)
    cv.imshow('final1', lung1)
    cv.imshow('final2', lung2)
    cv.imshow('mlung1', mlung1)
    cv.imshow('mlung2', mlung2)
    cv.waitKey()


def Method1(k):
    lung1, lung2, mlung1, mlung2, mask1, mask2 = Lungs(k)
    T1 = GetTripleOtsuFast(lung1, mask1)
    T2 = GetTripleOtsuFast(lung2, mask2)
    tlung1 = TripleThreshold(lung1, T1)
    tlung2 = TripleThreshold(lung2, T2)
    flung1 = Kth_filter(tlung1, (5, 5), 12)
    flung2 = Kth_filter(tlung2, (5, 5), 12)
    cv.imshow('lung1', lung1)
    cv.imshow('lung2', lung2)
    cv.imshow('mlung1', mlung1)
    cv.imshow('mlung2', mlung2)
    cv.imshow('flung1', flung1)
    cv.imshow('flung2', flung2)
    cv.waitKey()


def Method2(k):
    Original, Manuals, Lmasks = [Lungs(k)[0:2], Lungs(k)[2:4], Lungs(k)[4:6]]
    Tlungs = TLungs(Original, Lmasks)
    Slungs = SLungs(Tlungs, Lmasks)
    Flungs = [Kth_filter(Slungs[0], (5, 5), 12), Kth_filter(Slungs[1], (5, 5), 12)]
    cv.imshow('lung1', Original[0])
    cv.imshow('lung2', Original[1])
    cv.imshow('mlung1', Manuals[0])
    cv.imshow('mlung2', Manuals[1])
    cv.imshow('flung1', Flungs[0])
    cv.imshow('flung2', Flungs[1])
    cv.waitKey()


Method1(1)  # apply method 1 on image k
Method2(1)  # apply method 2 on image k
Method3(66)  # apply method 3 on image kk
