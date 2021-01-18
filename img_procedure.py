import numpy as np
from PIL import Image

def getImg(img_dir):
    try:
        #Karakterleri Gri tonlama ile alma.
        img = Image.open(img_dir).conver('L')
    except:
        return None

    #Resmi 25X25 yapma.
    if img.size !=(25,25):
        img=img.resize(25,25)

    #Resimlerin bulunduğu matris listesini geri dödürme
    img=np.array(img).tolist()
    img_list=[]
    for line in img:
        for value in line:
            img_list.append(value)
    return img_list

def getImgDir(argumans):

    #Resim dizinini alalım
    if len(argumans)<2:
        return None
    return argumans[1]


