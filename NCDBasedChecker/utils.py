from NCD import NCD
from Submission import Submission
import scipy.cluster.hierarchy as hc
import matplotlib.pyplot as plt
import os
from tokenizer import JavaTokenizer,PythonTokenizer
import numpy as np
from pyminifier import obfuscate

name_generator = obfuscate.obfuscation_machine(False,5)

def SubmissionArray_to_NCDMatrix(subs):
    matrix=[]
    for i in range(0,len(subs)):
        row=[]
        for j in range(0,len(subs)):
            if(i!=j):
                ncd_obj=NCD(subs[i].code.encode(),subs[j].code.encode())
                row.append(ncd_obj)
            else:
                row.append(0)
        matrix.append(row)
    return matrix

def methodArray_to_SubmissionArray(arr,student_info,mode=None,lang='java',obf_var=False):
    res= [Submission(tokenize(arr[i]['code'],mode,lang),student_info[i],i) for  i in range(len(arr))]
    return res

def fileNameArray_to_SubmissionArray(arr,student_info):
    return [Submission(arr[i],student_info[i],i) for  i in range(len(arr))]

def plot_dendrogram(linkage,level=25, mode='level'):
    plt.figure()
    R=hc.dendrogram(linkage,p=level, truncate_mode=mode, distance_sort=False)
    plt.show()
    return R
def getTokenVector(method,filterTokens=set()):
    os.chdir('./NCDBasedChecker')
    lexer = JavaTokenizer(filePath=None,code=method,filterTokens=filterTokens)
    res=lexer.getTokenVector()
    os.chdir('..')
    return lexer.getTokenVector()

def tokenize(method,mode, lang='java',obf_var=False):
    if lang=='java':
        if(not mode):
            return method
        os.chdir('./NCDBasedChecker')
        lexer = JavaTokenizer(filePath=None,code=method)
        tc,oc = lexer.getTokenizedCode()
        os.chdir('..')
        return tc if mode=='tok' else oc
    else:
        if(not mode):
            return method
        os.chdir('./NCDBasedChecker')
        lexer = PythonTokenizer(filePath=None,code=method,name_generator=name_generator)
        tc,oc = lexer.getTokenizedCode(obfuscate_var=obf_var)
        os.chdir('..')
        return tc if mode=='tok' else oc


def tokenizeFiles(arr,dir=None):
    parent_dir1="./test-files/TokenizedCode/"
    parent_dir2="./test-files/StrippedCode/"
    arr1=[]
    arr2=[]
    if not os.path.isdir(parent_dir1):
        os.mkdir(os.path.join(parent_dir1,""))
    if not os.path.isdir(parent_dir2):
        os.mkdir(os.path.join(parent_dir2,""))
    for file in arr:
        tokenizer=JavaTokenizer(file)
        tc,oc=tokenizer.getTokenizedCode()
        name=os.path.basename(file)
        
        with open(os.path.join(parent_dir1,name[0:len(name)-4]+"TOKENIZED.txt"),"w") as f:
            f.write(tc)
        arr1.append(os.path.abspath(os.path.join(parent_dir1,name[0:len(name)-4]+"TOKENIZED.txt")))
        with open(os.path.join(parent_dir2,name[0:len(name)-4]+"STRIPPED.txt"),"w") as f:
            f.write(oc)
        arr2.append(os.path.abspath(os.path.join(parent_dir2,name[0:len(name)-4]+"STRIPPED.txt")))
    return arr1,arr2
            
def deleteFiles():
    parent_dir1="./test-files/TokenizedCode"
    parent_dir2="./test-files/StrippedCode"
    for f in os.listdir(parent_dir1):
        if not f.endswith(".txt"):
            continue
        os.remove(os.path.join(parent_dir1, f))

    for f in os.listdir(parent_dir2):
        if not f.endswith(".txt"):
            continue
        os.remove(os.path.join(parent_dir2, f))

def sigmoid (x, mean=0.85):
    return 1/(1+np.exp(-30*(x-mean)))*100
    
