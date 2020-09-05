import os
import torch
import numpy as np
import torch
import json
from torch.utils.data import Dataset

np.random.seed(0)
class wsalDataset(Dataset):
    def __init__(self, datapath, mode, len_snippet, stream):
        print(f"Stream: {stream}")
        self.mode = mode
        self.len_snippet = len_snippet # max number of clips in video
        self.features = np.load(os.path.join(datapath,f"feature_{mode}.npy"), allow_pickle=True)[()]
        self.vnames = sorted(list(self.features.keys()))
        self._filter_feats(stream)
        dict_annot = json.load(open(os.path.join(datapath,"reduced_annotation.json")))
        self.cnames = dict_annot["list_classes"]

        self._filter_vnames(dict_annot["database"])

        assert sorted(self.vnames) == sorted(list(self.features.keys()))

        print(f"Mode: {mode}\nNumber of Videos:{len(self.vnames)}\nNumber of classes:{len(self.cnames)}")

        if self.mode == "val":
            self.set_ambiguous = dict_annot["set_ambiguous"]
            self.annts_cwise = self._get_annot_cwise(dict_annot["database"])
            self.len_snippet = max([feats.shape[0] for feats in self.features.values()])

        self.labels = self._get_labels(dict_annot["database"])
        self.fps_extracted = dict_annot["miscellany"]["fps_extracted"]
        self.len_feature_chunk = dict_annot["miscellany"]["len_feature_chunk"]
        self.vnames_cwise = self._get_vnames_cwise()
        
    def _filter_feats(self, stream):
        for vname in self.features:
            if stream == "rgb":
                self.features[vname] = self.features[vname][:,:1024]
            else:
                self.features[vname] = self.features[vname][:,1024:]

    def _filter_vnames(self, annts):
        vnames_string_print = '(vnames) %s: %d -> ' % (self.mode, len(self.vnames))
        feats_string_print = '(feats) %s: %d -> ' % (self.mode, len(self.features))
        vnames_filtered = []
        for v in self.vnames:
            if v not in annts or not annts[v]['annotations']:
                del self.features[v]
            else:
                vnames_filtered.append(v)

        self.vnames = vnames_filtered
        print (vnames_string_print + str(len(self.vnames)))
        print(feats_string_print + str(len(self.features)))

    def _get_labels(self, annts):
        num_class = len(self.cnames)
        labels = {}
        for v in self.vnames:
            labels[v] = np.zeros((num_class), dtype=np.float32)
            list_l = []
            for a in annts[v]['annotations']:
                list_l.append(a['label'])

            labels[v][[self.cnames.index(l) for l in set(list_l)]] = 1

        return labels
    
    def _get_vnames_cwise(self):
        vnames_cwise = []
        for c in self.cnames:
            vnames_cwise.append([v for v in self.vnames if self.labels[v][self.cnames.index(c)]])

        return vnames_cwise
    
    def _get_annot_cwise(self, annts):
        annot_cwise = {}
        for i,v in enumerate(self.vnames):
            for a in annts[v]["annotations"]:
                cn = a["label"]
                if cn not in annot_cwise:
                    annot_cwise[cn] = []
                annot_cwise[cn].append([i, a["segment"][0], a["segment"][1]])
        return annot_cwise
    
    def _preprocess(self, features):
        len_features = features.shape[0]
        if len_features >= self.len_snippet:
            start_idx = np.random.randint(len_features-self.len_snippet+1)
            return features[np.arange(start_idx,start_idx+self.len_snippet)], self.len_snippet
        else:
            return np.pad(features[np.arange(len_features)], ((0,self.len_snippet-len_features), (0,0)), mode='constant', constant_values=0), len_features

    def get_annts_classwise(self):
        return self.annts_cwise

    def get_set_ambiguous(self):
        return self.set_ambiguous

    def get_feature_rate(self):
        return self.fps_extracted / self.len_feature_chunk

    def get_cnames(self):
        return self.cnames

    def get_vnames(self):
        return self.vnames

    def get_num_classes(self):
        return len(self.cnames)

    def get_num_videos(self):
        return len(self.vnames)

    def __getitem__(self, i):
        v = self.vnames[i]
        features, len_features = self._preprocess(self.features[v])

        return features, self.labels[v], len_features, v
    
    def __len__(self):
        return len(self.vnames)

