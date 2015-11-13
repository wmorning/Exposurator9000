from __future__ import print_function, division
from sklearn import svm, cross_validation
import numpy as np

def confusion_plot(clf, X, y, plot=True):
    y_pred = clf.predict(X)
    label_mat = np.zeros((29,29))
    
    for i, label in enumerate(y):
        label_mat[label,y_pred[i]]+=1
    
    if plot==True:
        plt.imshow(label_mat)
        plt.xlabel(r'True Label')
        plt.ylabel(r'Predicted Label')
        plt.colorbar()
    
def diagnostic_vs_m_plot(X,y,nsteps=5):
    Xtrain, Xtest, ytrain, ytest = cross_validation.train_test_split(X,y,test_size=0.2)
    ii = ytrain.argsort()
    Xtrain = Xtrain[ii]
    ytrain = ytrain[ii]
    mtot = len(Xtrain)
    mtrain = np.arange(mtot/nsteps,mtot,mtot/nsteps)
    print(mtrain)
    tre = []
    tee = []
    confusion = np.ndarray((29,29))
    cfs = []
    
    for m in mtrain:
        clf = svm.SVC(kernel='linear')
        clf.fit(Xtrain[:m],ytrain[:m])
        ytrpred = clf.predict(Xtrain[:m])
        ytepred = clf.predict(Xtest)
        
        tre.append(len(np.where(ytrain[:m]!=ytrpred)[0])/len(ytrpred))
        tee.append(len(np.where(ytest!=ytepred)[0])/len(ytest))

        for i, label in enumerate(ytest):
            confusion[label-1,ytepred[i]-1]+=1
            
        cfs.append(confusion)
        confusion = np.zeros((29,29))
        
    return np.array(tre), np.array(tee), cfs, mtrain, clf
