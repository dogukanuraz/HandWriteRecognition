import sys
from os import listdir
from img_procedure import *
from classifier_procedure import *

def getDatasetsFromDir(datasets_dir):
    ##Bu işlev geri dönüş eğitimi veya veri kümelerindeki tüm dizinden girdi ve çıktı veri kümelerini test eder.
    #X eğitim verisi Y etiket.
    X_train, Y_train, X_test, Y_test = [],[],[],[]

    # Eğitim karakterleri almak:
    try:
        char_dirs = listdir(datasets_dir)
    except:
        return None,None

    # .Ds_Store karakter dizini olmadığı için siliyoruz:

    if '.DS_Store' in char_dirs:
        char_dirs.remove('.DS_Store')

    #Karakter Görüntülerinin boş olup olmadığını kontrol ediyoruz:
    for char_dir in char_dirs:
        if len(listdir(datasets_dir+'/'+char_dir)) <1:
            char_dirs.remove(char_dir)
        else:
            continue

    # Karakter listesi boş ise None döndürüyoruz:

    if len(char_dirs) <1:
        return None,None

    try:
        # Veri setini eğitmek için boş resimler ekliyoruz:
        emptyBlack = getImg('images/emptyBlack.png')
        X_train.append(getImg('images/emptyWhite.png'))
        X_train.append(getImg('images/emptyBlack.png'))
        Y_train.append(ord(' '))
        Y_train.append(ord(' '))
    except:
        print('Boş Resim(Eğitim için) Bulunamadı')

    #Tüm karakterlerden resim ve veri setleri alıyoruz:

    for char_dir in char_dirs:
        img_dirs = listdir(datasets_dir + '/' + char_dir)

        if '.DS_Store' in img_dirs:
            img_dirs.remove('.DS_Store')

        #Eğitim ve test verilerini bölüyoruz:
        point = int(0.9 * len(img_dirs))
        train_imgs, test_imgs = img_dirs[:point], img_dirs[point:]

        #Eğitim ve test verilerinin matrisini oluşturuyoruz:
        for img_dir in train_imgs:
            X_train.append(getImg(datasets_dir + '/' + char_dir + '/' + img_dir))
            Y_train.append(ord(char_dir))
        for img_dir in test_imgs:
            X_test.append(getImg(datasets_dir + '/' + char_dir + '/' + img_dir))
            Y_test.append(ord(char_dir))

    return X_train,Y_train,X_test,Y_test

def main():

    #image yolunu bulma
    imgDir = getImgDir(sys.argv)
    if imgDir is None:
        print('Resim dizini bulunamadı!')
        return

    #image bulma
    img = getImg(imgDir)
    if img is None:
        print('Resim bulunamadı!')
        return

    #Sınıflandırıcıyı kaydetme

    clf = getClassifier()

    #Sınıflandırıcıyı kaydetmediysek oluşturalım

    if clf is None:
        clf=createDecisionTree()
        X_train, Y_train, X_test, Y_test = getDatasetsFromDir('images/train')
        if X_train is None:
            print('Karakter veriseti  bulunamadı!')
            return

        clf = trainClassifier(clf, X_train, Y_train)

        saveClassifier(clf)

        print('Eğitim Puanı:', getScore(clf, X_train, Y_train))
        print('Test Puanı:', getScore(clf, X_test, Y_test))


    print('Tahmin:', chr(getPredict(clf, img)))
    return

if __name__ == '__main__':
    main()




