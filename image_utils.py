import sys
import scipy.io as sio
import json, itertools
import numpy as np

def_fnames='/users/ud2017/hoavt/data/flickr30k-cnn/flickr30k/filenames.json'
def_feats='/users/ud2017/hoavt/data/flickr30k-cnn/flickr30k/vgg_feats.mat'
class ImageFeatures(object):
    def __init__(self, names_files=def_fnames, feats_files=def_feats):
        self.cache = {}
        self.name2idx={}
        self.names_files = names_files
        #self.feats_files = feats_files
        with open(self.names_files,'rb') as fn:
            names=json.load(fn)
        fdata= sio.loadmat(feats_files)
        self.feats = np.array(fdata.get('feats')).T
        print self.feats.shape
        print self.feats[0]
        #for img_file, feat in itertools.izip(names,feats):
        #    self.cache[img_file] = feat
        #count=0
        for img_file in names:
            self.name2idx[img_file]= len(self.name2idx)
    
    def get_feat(self,img_file):
        if img_file in self.name2idx:
            return self.feats[self.name2idx[img_file]]
        return None


if __name__ == '__main__':
    images_feats = ImageFeatures()
    print images_feats.get_feat('3636329461.jpg')
