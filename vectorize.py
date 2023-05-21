from annoy import AnnoyIndex
from create_index import create_index
import torch
from sentence_transformers import SentenceTransformer,util
from intellidiff import IntelliDiff
import re
import os
import numpy as np
import json
from pathlib import Path
from scipy.spatial.distance import cdist

def load_json(file):
    file = _file_ending(file, "json")
    with open(file, 'r') as f:
        return json.load(f)

def save_json(obj: dict, file):
    file = _file_ending(file, "json")
    create_directories(file)
    with open(file, 'w') as f:
        json.dump(obj, f, indent=4)

def create_directories(path):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
          
def _file_ending(file, ending):
    return f"{file}.{ending}" if f".{ending}" not in file else file

def vectorize(path='./410_descr_methods',out_path='./410_descr_vectors.pt'):
    methods_410= load_json(path)
    mod=IntelliDiff()
    out=[]
    for i,method in enumerate(methods_410):
        if(i%100==0):
            print(f"{i+1}/{len(methods_410)}")
        sents=mod.getSentences(method['description'])
        embs=mod.getSentenceEmbeddings(sents)
        if(len(embs)>1):
            out.append(np.sum(np.concatenate(embs,axis=0),axis=0,keepdims=True))
        else:
            out.append(embs[0])

    #np.save(np.concatenate(out,axis=0),out_path)
    return np.concatenate(out,axis=0)
    #torch.save(torch.from_numpy(np.concatenate(out,axis=0)),out_path)

def generate_sim_matrix(path='./410_descr_methods'):
    methods= load_json(path)
    res= np.zeros((len(methods),len(methods)))
    mod=IntelliDiff()
    embs=[]
    for i,method in enumerate(methods):
        if(i%100==0):
            print(f"{i+1}/{len(methods)}")
        sents=mod.getSentences(method['description'])
        embs.append(mod.getSentenceEmbeddings(sents))
    for i in range(res.shape[0]):
        mat_a = np.array(embs[i]).reshape((len(embs[i]),-1))
        for j in range(i+1, res.shape[0]):
            mat_b = np.array(embs[j]).reshape((len(embs[j]),-1))
            sims= cdist(mat_a,mat_b)
            axis= 1 if sims.shape[0]<=sims.shape[1] else 0
            sim=np.mean(np.max(sims,axis=axis))
            res[i,j]=sim
            res[j,i]=sim
        
    return (res-np.min(res))/(np.max(res)-np.min(res))

#vectorize()
