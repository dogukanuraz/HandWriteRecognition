from sklearn.externals import joblib
from sklearn import tree

def createDecisionTree():
    #Sınıflandırıcı oluşturuyoruz.
    return tree.DecisionTreeClassifier()

def getClassifier():
    #Eğitilmiş Sınıflandırıcıyı alıyoruz.
    try:
        clf=joblib.load('classifier/classifier.pkl')
    except:
        return None
    return clf

def trainClassifier(clf,X,Y):
    #Eğitim Sınıflandırıcısı
    return clf.fit(X,Y)

def getScore(clf,X,Y):
    return clf.score(X,Y)

def getPredict(clf,img):
    #Tahmin al
    return clf.predict(img)

def saveClassifier(clf):
    #Sınıflandırıcıyı kaydet.
    joblib.dump(clf,'classifier/classifier.pkl')