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
    
def diagnostic_vs_m(X,y,nsteps=5):
    Xtrain, Xtest, ytrain, ytest = cross_validation.train_test_split(X,y,test_size=0.2)

    #fudge to give more than 1 class for small training sets
    uy = np.unique(ytrain)
    print('training labels: {0}'.format(uy))


    mtot = len(Xtrain)
    print('mtot: {0}'.format(mtot))
    print('mtest: {0}'.format(len(ytest)))
    print('farts: {0}'.format(len(np.where(y!=29))/len(y)))
    mtrain = np.logspace(1,np.log10(mtot),nsteps)
    mtrain = np.floor(mtrain)
    print(mtrain)
    tre = []
    tee = []
    confusion = np.zeros((29,29))
    cfs = []

    muy = np.unique(ytrain[:mtrain[0]])
    if len(muy)==1:
        ytrain[0] = muy[0]-1
    
    for m in mtrain:
        clf = svm.SVC(kernel='linear')
        clf.fit(Xtrain[:m],ytrain[:m])
        ytrpred = clf.predict(Xtrain[:m])
        ytepred = clf.predict(Xtest)

        ntr = len(np.where(ytrain[:m]!=ytrpred)[0])
        nte = len(np.where(ytest!=ytepred)[0])
        
        print('Number of training errors with {0} examples: {1}'.format(m,ntr))
        print('Number of test errors with {0} examples: {1}'.format(m,nte))
        
        tre.append(ntr/len(ytrpred))
        tee.append(nte/len(ytest))

        for i, label in enumerate(ytest):
            confusion[label-1,ytepred[i]-1]+=1
            
        cfs.append(confusion)
        confusion = np.zeros((29,29))
        
    return np.array(tre), np.array(tee), cfs, mtrain, clf
