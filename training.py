from __future__ import division, print_function
from glob import glob
from sklearn import svm, cross_validation
import numpy as np
import plot_utils as pu
import json
import sys
import fitsio
import pickle
import os

class Artifact(object):
    
    def __init__(self, identifier, expname, ccd, problem, x, y):
        self.ident = identifier
        self.expname = expname
        self.ccd = ccd
        self.problem = problem
        self.x = x
        self.y = y
        

def load_release_artifacts(artifact_base):
    files = glob(artifact_base+'*')
    artifacts = []
    for f in files:
        with open(f, 'r') as fp:
            arts = json.load(fp)
            for art in arts:
                if ~art['false_positive']:
                    ident = '_'.join([art['expname'].split('_')[-1],str(art['ccd'])])
                    oart = Artifact(ident, art['expname'], art['ccd'],\
                                    art['problem'], art['x'], \
                                    art['y'])
                    artifacts.append(oart)

    artifacts.sort(key=lambda x : x.ident)
    artifacts = np.array(artifacts)
    
    return artifacts

def coarsegrain(stamp, factor=8):
    nx, ny = stamp.shape
    cnx = nx//factor
    cny = ny//factor
    
    cgstamp = np.ndarray((cnx, cny))
    for i in np.arange(factor-1):
        stamp[i+1::factor,:] += stamp[i::factor,:]
    for j in np.arange(factor-1):
        stamp[:,j+1::factor] += stamp[:,j::factor]
        
    cgstamp = stamp[factor-1::factor,factor-1::factor]/float(factor**2)
    return(cgstamp)
    

def create_design_matrix(imagenames, bkgnames, artifacts, gridsize=128, cgfactor=8, farts=0.5, save_mb=False):
    assert((2048%gridsize==0) & (gridsize<=2048))
    assert(len(imagenames)==len(bkgnames))
    nxpixels  = 4096
    nypixels = 2048
    nxgridelem = nxpixels//gridsize
    nygridelem = nypixels//gridsize
    
    if len(imagenames)==1:
        edirs = np.array(glob(imagenames))
        bdirs = np.array(glob(bkgnames))
    else:
        edirs = imagenames
        bdirs = bkgnames

    eident = []
    bident = []
    for i, d in enumerate(edirs):
        if d=='':continue
        print(bdirs[i])
        eexpnum = d.split('/')[-1].split('_')
        eexpnum[2] = eexpnum[2].split('.')[0]
        bexpnum = bdirs[i].split('/')[-1].split('_')
        assert('_'.join(eexpnum[1:3])==('_'.join(bexpnum[1:3])))
        eident.append('_'.join(eexpnum[1:3]))
        bident.append('_'.join(bexpnum[1:3]))
    
    aident = np.array([a.ident for a in artifacts], dtype=str)
    print(aident)
    print(aident.dtype)
    aident = np.sort(aident)
    print(aident.dtype)
    eident = np.array(eident)
    bident = np.array(bident)
    assert(len(eident)==len(bident))
    
    eidx = eident.argsort()
    bidx = bident.argsort()
    
    eident = eident[eidx]
    edirs = edirs[eidx]
    bident = bident[bidx]
    bdirs = bdirs[bidx]
    
    features = []
    labels = []
    xgrid = np.arange(0,nxpixels-1,gridsize)
    ygrid = np.arange(0,nypixels-1,gridsize)
    print('xgrid: {0}'.format(len(xgrid)))
    print('ygrid: {0}'.format(len(ygrid)))
    print(eident.dtype)
    print(aident.dtype)
    mb = 0
    aidx = np.searchsorted(aident, eident[0])
    for i, e in enumerate(eident):
        if savemb:
            if (len(features)%100==0) & (len(features)!=0):
                features = np.array(features)
                labels = np.array(labels)
                np.save('/nfs/slac/g/ki/ki23/des/jderose/des/inDianajonES/data/X_{0}_{1}_{2}_{3}_mb{4}.npy'.format(len(eident), farts, gridsize, cgfactor, mb), features)
                np.save('/nfs/slac/g/ki/ki23/des/jderose/des/inDianajonES/data/y_{0}_{1}_{2}_{3}_mb{4}.npy'.format(len(eident), farts, gridsize, cgfactor, mb), labels)
                features = []
                labels = []
                mb+=1

                
        assert(e==bident[i])
        print(e)
        exp = fitsio.read(edirs[i], ext=1)
        msk = fitsio.read(edirs[i], ext=2)
        #wgt = fitsio.read(edirs[i], ext=3) maybe use weights later
        bkg = fitsio.read(bdirs[i], ext=1)

        mbse = (exp-bkg)
        mbse[msk>np.min(msk)] = -99
        ac = []
        aa = []
        for j, xb in enumerate(xgrid):
            for k, yb in enumerate(ygrid):
                if (j>(len(xgrid)-1)) or (k>(len(ygrid)-1)): continue
                stamp = coarsegrain(mbse[xb:(xb+gridsize),yb:(yb+gridsize)], cgfactor).flatten()
                features.append(stamp)
                tidx = aidx
                problems = []
                while(str(artifacts[tidx].ident)==str(e)):
                    a = artifacts[tidx]
                    if (a not in ac) and (a not in aa):
                        ac.append(a)
                    #Include artifacts near the edges of 
                    #the grid in all pixels near that edge?
                    if (((xb<=a.x) & (a.x<(xb+gridsize))) &\
                        ((yb<=a.y) & (a.y<(yb+gridsize)))):
                        print('Bingo!')
                        problems.append(a.problem)
                        aa.append(a)

                    if tidx==(len(artifacts)-1):
                        break
                    else:
                        tidx+=1

                #for now, keep only one artifact per grid element
                if len(problems)>0:
                    labels.append(problems[0])
                else:
                    labels.append('no_artifacts')
        if len(ac)!=len(aa):
            print('Unassigned artifacts: {0}, {1}'.format(ac,aa))
            print(ac[0].ident)
            print(ac[0].x)
            print(ac[0].y)

        aidx+=(tidx-aidx)
        
    features = np.array(features)
    labels = np.array(labels)

    if save_mb:
        np.save('/nfs/slac/g/ki/ki23/des/jderose/des/inDianajonES/data/X_{0}_{1}_{2}_{3}_mb{4}.npy'.format(len(eident), farts, gridsize, cgfactor, mb), features)
        np.save('/nfs/slac/g/ki/ki23/des/jderose/des/inDianajonES/data/y_{0}_{1}_{2}_{3}_mb{4}.npy'.format(len(eident), farts, gridsize, cgfactor, mb), labels)
    
    return features, labels

def enumerate_labels(labels):
    atype = ['Column mask', 'Cosmic ray', 'Cross-talk', 'Edge-bleed', 'Excessive mask', 'Dark rim',
         'Dark halo', 'Quilted sky', 'Wavy sky', 'Anti-bleed', 'A/B jump', 'Fringing', 'Tape bump',
         'Tree rings', 'Vertical jump', 'Ghost', 'Bright spray', 'Brush strokes', 'Bright arc',
         'Satellite', 'Airplane', 'Guiding', 'Shutter', 'Readout', 'Haze', 'Vertical stripes',
         'Other...', 'Awesome!', 'no_artifacts']
    nums = range(1,len(atype)+1)
    ldict = dict(zip(atype,nums))
    enumlabels = np.array([ldict[l] for l in labels])
    
    return enumlabels

def get_unrepresentative_training(runs, expids, accds, aidents, nimg=None, farts=0.5):

    basedir = '/nfs/slac/g/ki/ki21/cosmo/DESDATA/OPS/red'
    aimgnames = []
    abkgnames = []
    nimgnames = []
    nbkgnames = []
    idents = []

    narts = len(np.unique(aidents))

    if (nimg!=None) and (narts/farts<nimg):
        print('Not enough artifacts to satisfy artifact fraction and nimg')
        return
    elif (nimg==None):
        nimg = narts/farts
        nnull = nimg-narts
    else:
        narts = nimg*farts
        nnull = nimg-narts

    print('nimg, nnull, narts: {0}, {1}, {2}'.format(nimg, nnull, narts))

    artcount = 0
    nullcount = 0

    for i, (run, expid, ccd, ident) in enumerate(zip(runs,expids,accds,aidents)):
        aim = '{0}/{1}/red/DECam_00{2}/DECam_00{2}_{3}.fits.fz'.format(basedir,run,expid,ccd)
        im = glob('{0}/{1}/red/DECam_00{2}/DECam_00{2}_*[0-9].fits.fz'.format(basedir,run,expid))
        abkg = '{0}/{1}/red/DECam_00{2}/DECam_00{2}_{3}_bkg.fits.fz'.format(basedir,run,expid,ccd)
        bkg = glob('{0}/{1}/red/DECam_00{2}/DECam_00{2}_*_bkg.fits.fz'.format(basedir,run,expid))
        im = [ti for ti in im if (os.path.isfile(ti) and ti!=aim)]
        bkg = [tb for tb in bkg if (os.path.isfile(tb) and tb!=abkg)]

        if os.path.isfile(aim) and os.path.isfile(abkg):
            aimgnames.append(aim)
            abkgnames.append(abkg)
            idents.append(ident)

        if len(im)!=len(bkg):continue
        nimgnames.extend(im)
        nbkgnames.extend(bkg)

    aimgnames, uii = np.unique(np.array(aimgnames), return_index=True)
    abkgnames = np.array(abkgnames)[uii]
    aidents = np.array(idents)[uii]

    nimgnames, uii = np.unique(np.array(nimgnames), return_index=True)
    nbkgnames  = np.array(nbkgnames)[uii]

    aidx = np.random.choice(np.arange(len(aimgnames)), size=narts, replace=False)
    nidx = np.random.choice(np.arange(len(nimgnames)), size=nnull, replace=False)
    aimgnames = aimgnames[aidx]
    abkgnames = abkgnames[aidx]
    aidents = aidents[aidx]
    nimgnames = nimgnames[nidx]
    nbkgnames = nbkgnames[nidx]

    return aimgnames, abkgnames, nimgnames, nbkgnames, aidents
    

def get_training_filenames(runs, expids, nimg=None):

    basedir = '/nfs/slac/g/ki/ki21/cosmo/DESDATA/OPS/red'
    imgnames = []
    bkgnames = []

    for run, expid in zip(runs,expids):
        im = glob('{0}/{1}/red/DECam_00{2}/DECam_00{2}_*[0-9].fits.fz'.format(basedir,run,expid))
        bk = glob('{0}/{1}/red/DECam_00{2}/DECam_00{2}_*_bkg.fits.fz'.format(basedir,run,expid))
        if len(im)!=len(bk):continue
        imgnames.extend(im)
        bkgnames.extend(bk)

    imgnames = np.unique(np.array(imgnames))
    bkgnames = np.unique(np.array(bkgnames))

    if nimg!=None:
        idx = np.random.choice(np.arange(len(imgnames)), size=nimg, replace=False)
        imgnames = imgnames[idx]
        bkgnames = bkgnames[idx]
    
    return imgnames, bkgnames

def train_and_validate(runs, expids, ccds, ident, artpath, nimg=None, farts=None, gridsize=128, cgfactor=8, store_design=False):

    artifacts = load_release_artifacts(artpath)
    aident = np.array([a.ident[2:] for a in artifacts], dtype=str)
    ident = np.array(ident, dtype=str)
    sii = np.argsort(ident)
    ident = ident[sii]
    runs = runs[sii]
    expids = expids[sii]
    ccds = ccds[sii]
    aidx = np.array([np.searchsorted(ident, idn) for idn in aident])
    aidx[aidx==len(ident)] = -1
    aidx = aidx[ident[aidx]==aident]
    print(aidx)
    print(len(np.unique(aidx)))
    
    print('Selecting training with fraction of artifacts: {0}'.format(farts))
    aimgnames, abkgnames, nimgnames, nbkgnames, mident = get_unrepresentative_training(runs[aidx], expids[aidx], ccds[aidx], ident[aidx], nimg=nimg, farts=farts)
    imgnames = np.hstack([aimgnames,nimgnames])
    bkgnames = np.hstack([abkgnames,nbkgnames])

    aii = np.argsort(aident)
    aident = aident[aii]
    artifacts = artifacts[aii]
    aidx = np.unique(np.array([np.searchsorted(aident, m) for m in mident]))

    X, y = create_design_matrix(imgnames, bkgnames, artifacts[aidx], gridsize=gridsize, cgfactor=cgfactor, farts=farts, save_mb=store_design)
    ey = enumerate_labels(y)
    
    if store_design:
        np.save('/nfs/slac/g/ki/ki23/des/jderose/des/inDianajonES/data/X_{0}_{1}_{2}_{3}.npy'.format(nimg, farts, gridsize, cgfactor), X)
        np.save('/nfs/slac/g/ki/ki23/des/jderose/des/inDianajonES/data/y_{0}_{1}_{2}_{3}.npy'.format(nimg, farts, gridsize, cgfactor), ey)

    tre, tee, cfs, mtrain, clf = pu.diagnostic_vs_m(X, ey, nsteps=5)

    edtype = np.dtype([('tre', tre.dtype), ('tee', tee.dtype), ('mtrain', mtrain.dtype)])
    print('tre: {0}'.format(tre))
    print('tee: {0}'.format(tee))
    error_data = np.ndarray(len(tre), dtype=edtype)
    error_data['tre'] = tre
    error_data['tee'] = tee
    error_data['mtrain'] = mtrain

    fitsio.write('error_data.{0}.fits'.format(nimg), error_data)
    with open('confusion.{0}.{1}.p'.format(nimg,farts), 'w') as fp:
        pickle.dump(cfs, fp)
    with open('clf.{0}.{1}.p'.format(nimg, farts), 'w') as fp:
        pickle.dump(clf, fp)

    return tre, tee, cfs, mtrain, clf


if __name__=='__main__':

    artpath = sys.argv[1]
    nimg = int(sys.argv[2])
    fart = float(sys.argv[3])
    gs = int(sys.argv[4])
    cgf = int(sys.argv[5])

    
    runinfo = np.genfromtxt('/u/ki/jderose/ki23/des/se_exposures/exp_run_info.csv', dtype=None, delimiter=',', skip_header=1)
    ident = np.array(['_'.join([str(r['f5']),str(r['f4'])]) for r in runinfo])
    tre, tee, cfs, mtrain, clf = train_and_validate(runinfo['f2'], runinfo['f5'], runinfo['f4'], ident, artpath, nimg=nimg, farts=fart, gridsize=gs, cgfactor=cgf, store_design=True)
    
