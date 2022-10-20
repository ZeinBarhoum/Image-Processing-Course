import time

import cv2 as cv
import numpy as np


Num = 100
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


def GetHistogram(img, mask, a, b):
    nop = np.zeros(shape=256, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (mask[i][j] != 0 and img[i][j] >= a and img[i][j] <= b):
                nop[img[i][j]] += 1

    return nop


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


def Lungs(k):
    image = GetImage(k)
    image=cv.GaussianBlur(image,(5,5),0)
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


def rotate_image(image, angle, image_center):
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, (1024, 1024), flags=cv.INTER_LINEAR)
    cv.waitKey()
    return result


def angel(a, b):
    return 90 - (np.arctan2(a[0] - b[0], a[1] - b[1])) * 180 / np.pi


def GetMiddle(img):
    xall, yall = np.where(img != 0)
    x = sum(xall) / len(xall)
    y = sum(yall) / len(yall)
    return [x, y]


def dis(a, b):
    return np.sqrt(np.power(a[0] - b[0], 2) + np.power(a[1] - b[1], 2))


StoreAll()

np.save('GlobalLungs1', GlobalLungs1)
np.save('GlobalLungs2', GlobalLungs2)
np.save('GlobalMLungs1', GlobalMLungs1)
np.save('GlobalMLungs2', GlobalMLungs2)
np.save('GlobalMasks1', GlobalMasks1)
np.save('GlobalMasks2', GlobalMasks2)
np.save('GlobalDis1', GlobalDis1)
np.save('GlobalDis2', GlobalDis2)
np.save('GlobalAnngles1', GlobalAnngles1)
np.save('GlobalAnngles2', GlobalAnngles2)
np.save('Globalang1', Globalang1)
np.save('Globalang2', Globalang2)
np.save('GlobalThresholds1', GlobalThresholds1)
np.save('GlobalThresholds2', GlobalThresholds2)
