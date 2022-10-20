

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os 

def GetDft(img):
    f=np.fft.fft2(img)
    return f
def CenterDft(f):
    f1=np.copy(f)
    f1=np.fft.fftshift(f)
    return f1
def normalize(img):
    min=np.min(img)
    max=np.max(img)
    img1=np.copy(img)
    img1=np.round((img1-min)*255/(max-min))
    img1=img1.astype(np.uint8)
    return img1
def GetDftImage(f,layer='mag',type='power',par=1):
    f1 = np.copy(f)
    if(layer=='mag'):
        f1=np.abs(f1)
    elif(layer=='phase'):
        f1=np.abs(f1)

    if(type=='log'):
        f1=np.log(1+f1)
    elif(type=='power'):
        f1=np.power(f1,par)
    f1=normalize(f1)
    return f1
def GetIdft(f):
    img=np.fft.ifft2(f)
    img=normalize(np.abs(img))
    img=img.astype(np.uint8)
    return img
def GetLoG(size,sigma,Max_wanted=-1):
    f=np.zeros(shape=size,dtype=float)  #Create the filter with zero values
    midx=int(size[0]/2)  #x of the center
    midy=int(size[1]/2)  #y of the center
    for x in range(size[0]):  #iterate over the filter
        for y in range(size[1]):
            #Calculate the LoG(x,y)
            x2=1.0*np.power(x-midx,2)
            y2=1.0*np.power(y-midy,2)
            s2=1.0*np.power(sigma,2)
            exp=1.0*np.exp(-1.0*(x2+y2)/(2*s2))
            k=1.0*(x2+y2-2*s2)/(np.pi*np.power(sigma,6))
            f[x][y]=k*exp

    f[midx][midy]-=sum(sum(f)) #the sum of all values should be 0
    f=f*Max_wanted/(f[midx][midy])
    return f
def pad(img,px,py):
    img1=np.zeros(shape=(px,py),dtype=img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img1[i][j]=img[i][j]
    return img1
def ApplyKernal(img,f):
    px = img.shape[0] + f.shape[0]-1
    py = img.shape[1] + f.shape[1]-1
    pimg=pad(img,px,py)
    pfil=pad(f,px,py)
    ftimg=GetDft(pimg)
    ftfil=GetDft(pfil)
    ftnew=np.multiply(ftimg,ftfil)
    pnimg=GetIdft(ftnew)
    nimg=np.zeros(img.shape,dtype=img.dtype)
    for i in range(nimg.shape[0]):
        for j in range(nimg.shape[1]):
            nimg[i][j] = pnimg[i + int((f.shape[0]-1)/2)][j + int((f.shape[1]-1)/2)]
    return nimg
def ApplyLoG(img,size,sigma):
    f = GetLoG(size, sigma, 1)
    filterdImg = ApplyKernal(img, f)
    float_img = img.astype(float)
    float_filterd = filterdImg.astype(float)
    addition = float_img + float_filterd
    addition = normalize(addition)
    return addition
def GenerateLPF_HPF(type,function,size,center,d0=0,deg=0,Centerkeep=0):
    fil=np.zeros(shape=size,dtype=np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            d=np.sqrt(np.power(i-center[0],2)+np.power(j-center[1],2))
            if function=='ideal':
                fil[i][j]=255*int(d<=d0)
            elif function=='bw':
                fil[i][j]=int(np.round(255.0/(1+np.power(d/d0,2*deg))))
            elif function=='gs':
                fil[i][j]=int(np.round(255.0*np.exp(-np.power(d,2)/(2*np.power(d0,2)))))

            if type=='high':
                fil[i][j]=255-fil[i][j]
    if Centerkeep==1:
        fil[center[0]][center[1]]=255
    return fil
def ApplyFreqFilter(img,f):
    ft=GetDft(img)
    ft=CenterDft(ft)
    nft = np.copy(ft)
    for i in range(nft.shape[0]):
        for j in range(nft.shape[1]):
            nft[i][j]=ft[i][j]*(float(f[i][j]))/255.0

    nimg=GetIdft(CenterDft(nft))
    return nimg
def main1():
    currentPath=os.path.dirname(os.path.abspath(__file__))
    
    img=cv.imread('Girl.tif',0)
    cv.imshow('img',img)
    cv.waitKey(0)
    img2=ApplyLoG(img,(7,7),1)
    img3 = ApplyLoG(img, (7, 7), 1)
    cv.imshow('original', img)
    cv.imwrite(os.path.join(currentPath,'originalGirl.jpg'), img)
    cv.imshow('New1', img2)
    cv.imwrite(os.path.join(currentPath,'NewGirl1.jpg'), img2)
    cv.imshow('New2', img3)
    cv.imwrite(os.path.join(currentPath,'NewGirl2.jpg'), img3)
def main2():

    img = cv.imread('Girl.tif', 0)
    fil1=GenerateLPF_HPF('low','ideal',img.shape,(int(img.shape[0]/2),int(img.shape[1]/2)),40)
    nimg1=ApplyFreqFilter(img,fil1)

    fil2 = GenerateLPF_HPF('low', 'gs', img.shape, (int(img.shape[0] / 2), int(img.shape[1] / 2)), 40)
    nimg2=ApplyFreqFilter(img, fil2)

    fil3 = GenerateLPF_HPF('low', 'bw', img.shape, (int(img.shape[0] / 2), int(img.shape[1] / 2)), 40,2)
    nimg3 = ApplyFreqFilter(img, fil3)

    fil4 = GenerateLPF_HPF('high', 'ideal', img.shape, (int(img.shape[0] / 2), int(img.shape[1] / 2)), 40,Centerkeep=1)
    nimg4 = ApplyFreqFilter(img, fil4)

    fil5 = GenerateLPF_HPF('high', 'gs', img.shape, (int(img.shape[0] / 2), int(img.shape[1] / 2)), 40,Centerkeep=1)
    nimg5 = ApplyFreqFilter(img, fil5)

    fil6 = GenerateLPF_HPF('high', 'bw', img.shape, (int(img.shape[0] / 2), int(img.shape[1] / 2)), 40, 2,Centerkeep=1)
    nimg6 = ApplyFreqFilter(img, fil6)


    cv.imshow('original',img)
    cv.imshow('filter1',fil1)
    cv.imshow('new image 1',nimg1)
    cv.imshow('filter2',fil2)
    cv.imshow('new image 2',nimg2)
    cv.imshow('filter3', fil3)
    cv.imshow('new image 3', nimg3)
    cv.imshow('filter4', fil4)
    cv.imshow('new image 4', nimg4)
    cv.imshow('filter5', fil5)
    cv.imshow('new image 5', nimg5)
    cv.imshow('filter6', fil6)
    cv.imshow('new image 6', nimg6)
    cv.waitKey()
def main3():
    img1=cv.imread('Clown.tif',0)
    ft1=GetDftImage(CenterDft(GetDft(img1)),'mag','log')
    fil1_1 = GenerateLPF_HPF('low', 'ideal', img1.shape, (121,189), 10)
    fil1_2 = GenerateLPF_HPF('low', 'ideal', img1.shape, (170, 104), 10)
    fil1_3 = GenerateLPF_HPF('low', 'ideal', img1.shape, (134, 125), 10)
    fil1_4 = GenerateLPF_HPF('low', 'ideal', img1.shape, (159, 167), 10)
    fil1=fil1_1+fil1_2+fil1_3+fil1_4
    fil1=255-fil1
    img2=ApplyFreqFilter(img1,fil1)
    cv.imshow('original',img1)
    cv.imshow('ft', ft1)
    cv.imshow('filter', fil1)
    cv.imshow('modified',img2)
    cv.waitKey()
    plt.subplot(221),plt.imshow(img1,cmap='gray')
    plt.title('original'),plt.xticks([]),plt.yticks([])
    plt.subplot(222), plt.imshow(ft1,cmap='gray')
    plt.title('ft'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(img2, cmap='gray')
    plt.title('img2'), plt.xticks([]), plt.yticks([])
    plt.show()
def main4():
    img = cv.imread('Halftone.tif', 0)
    ft = GetDftImage(CenterDft(GetDft(img)), 'mag', 'log')
    fil1 = GenerateLPF_HPF('low', 'ideal', img.shape, (int(img.shape[0]/2),int(img.shape[1]/2)), 50)
    fil2 = GenerateLPF_HPF('low', 'gs', img.shape, (int(img.shape[0] / 2), int(img.shape[1] / 2)), 50)
    fil3 = GenerateLPF_HPF('low', 'bw', img.shape, (int(img.shape[0] / 2), int(img.shape[1] / 2)), 50,2)
    fil4 = GenerateLPF_HPF('high', 'ideal', img.shape, (int(img.shape[0] / 2), int(img.shape[1] / 2)),256)
    filt5=fil3+fil4
    img1 = ApplyFreqFilter(img, fil1)
    img2 = ApplyFreqFilter(img, fil2)
    img3 = ApplyFreqFilter(img, fil3)
    img5 = ApplyFreqFilter(img, filt5)

    ft3 = GetDftImage(CenterDft(GetDft(img3)), 'mag', 'log')
    cv.imshow('original', img)
    cv.imshow('ft', ft)
    cv.imshow('img1', img1)
    cv.imshow('img2', img2)
    cv.imshow('img3', img3)
    cv.imshow('img5', img5)
    cv.imshow('filter5', filt5)
    cv.waitKey()
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(ft, cmap='gray')
    plt.title('ft'), plt.xticks([]), plt.yticks([])
    plt.show()
def main5():
    img = cv.imread('Clock.tif', 0)
    ft = GetDftImage(CenterDft(GetDft(img)), 'mag', 'log')
    fil1 = GenerateLPF_HPF('low', 'gs', img.shape, (int(img.shape[0] / 2), int(img.shape[1] / 2)), 35)
    img1 = ApplyFreqFilter(img, fil1)
    cv.imshow('original', img)
    cv.imshow('new', img1)
    cv.imshow('ft',ft)
    cv.imshow('filter', fil1)
    cv.waitKey()
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(ft, cmap='gray')
    plt.title('ft'), plt.xticks([]), plt.yticks([])
    plt.show()
def main6():
    img=cv.imread('Membrane.tif',1)
    img_b,img_g,img_r=cv.split(img)
    ft1 = GetDftImage(CenterDft(GetDft(img_b)), 'mag', 'log')
    ft2 = GetDftImage(CenterDft(GetDft(img_g)), 'mag', 'log')
    ft3 = GetDftImage(CenterDft(GetDft(img_r)), 'mag', 'log')
    cv.imshow('ft b',ft1)
    cv.imshow('ft g',ft2)
    cv.imshow('ft r',ft3)
    fil1 = GenerateLPF_HPF('low', 'gs', img_b.shape, (int(img_b.shape[0] / 2), int(img_b.shape[1] / 2)), 150)
    img_b2=ApplyFreqFilter(img_b,fil1)
    img_g2 = ApplyFreqFilter(img_g, fil1)
    img_r2 = ApplyFreqFilter(img_r, fil1)
    img2=cv.merge((img_b2,img_g2,img_r2))
    cv.imshow('1',img)
    cv.imshow('2',img2)
    cv.waitKey()
def main():
    f = GetLoG((7, 7), 1, -50)
    f = np.round(f).astype(int)
    print(f)

    img = cv.imread('smoothedclown8g.tif', 0)
    nimg1 = ApplyLoG(img, (7, 7), 1)
    nimg2= ApplyLoG(img, (7, 7), 1.5)
    nimg3= ApplyLoG(img, (7, 7), 0.5)
    cv.imshow('original', img)
    cv.imshow('filtered1', nimg1)
    cv.imshow('filtered2', nimg2)
    cv.imshow('filtered3', nimg3)
    cv.waitKey()


main1()