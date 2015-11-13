from __future__ import division, print_function
from glob import glob
from sklearn import svm, cross_validation
import plot_utils as pu
import json
import fitsio

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
    

def create_design_matrix(imagenames, backgroundnames, artifacts, gridsize=128, cgfactor=8):
    
    assert((2048%gridsize==0) & (gridsize<=2048))
    nxpixels  = 4096
    nypixels = 2048
    nxgridelem = nxpixels/gridsize
    nygridelem = nypixels/gridsize
    
    if len(imagenames)==1:
        edirs = np.array(glob(imagenames))
        bdirs = np.array(glob(bkgnames))
    else:
        edirs = imagenames
        bdirs = bkgnames

    eident = []
    bident = []
    for i, d in enumerate(edirs):
        eexpnum = d.split('/')[-1].split('_')
        eexpnum[2] = eexpnum[2].split('.')[0]
        bexpnum = bdirs[i].split('/')[-1].split('_')
        eident.append('_'.join(eexpnum[1:3]))
        bident.append('_'.join(bexpnum[1:3]))
    
    aident = np.array([a.ident for a in artifacts])
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
    imgcache = None
    xgrid = np.arange(0,nxpixels-1,gridsize)
    ygrid = np.arange(0,nypixels-1,gridsize)
    
    aidx = np.searchsorted(aident, eident[0])
    for i, e in enumerate(eident):
        assert(e==bident[i])
        exp = fitsio.read(edirs[i], ext=1)
        msk = fitsio.read(edirs[i], ext=2)
        #wgt = fitsio.read(edirs[i], ext=3) maybe use weights later
        bkg = fitsio.read(bdirs[i], ext=1)
        mbse = (exp-bkg)
        mbse[msk>np.min(msk)] = -99

        for j, xb in enumerate(xgrid):
            for k, yb in enumerate(ygrid):
                if (j>(len(xgrid)-1)) or (k>(len(ygrid)-1)): continue
                stamp = coarsegrain(mbse[xb:(xb+gridsize),yb:(yb+gridsize)], cgfactor).flatten()
                features.append(stamp)
                tidx = aidx
                problems = []
                while(str(artifacts[tidx].ident)==str(e)):
                    a = artifacts[tidx] 
                    #Include artifacts near the edges of 
                    #the grid in all pixels near that edge?
                    if (((xb<=a.x) & (a.x<(xb+gridsize))) &\
                        ((yb<=a.y) & (a.y<(yb+gridsize)))):
                        problems.append(a.problem)
                    tidx+=1
                #for now, keep only one artifact per grid element
                if len(problems)>0:
                    labels.append(problems[0])
                else:
                    labels.append('no_artifacts')

        aidx+=(tidx-aidx)
    
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels

def enumerate_labels(labels):
    atype = ['Column_mask', 'Cosmic_ray', 'Cross-talk', 'Edge-bleed', 'Excessive_mask', 'Dark_rim',
         'Dark_halo', 'Quilted_sky', 'Wavy_sky', 'Anti-bleed', 'AB_jump', 'Fringing', 'Tape_bump',
         'Tree_rings', 'Vertical_jump', 'Ghost', 'Bright_spray', 'Brush_strokes', 'Bright_arc',
         'Satellite', 'Airplane', 'Guiding', 'Shutter', 'Readout', 'Haze', 'Vertical_stripes',
         'Other...', 'Awesome!', 'no_artifacts']
    nums = range(1,len(atype)+1)
    ldict = dict(zip(atype,nums))
    enumlabels = np.array([ldict[l] for l in labels])
    
    return enumlabels

#def get_training_filenames(runs, expids)
