import lzma
import gzip
import numpy as np
from multiprocessing import Pool
import multiprocessing

class BlockNCD:

    def __init__(self,listOfSubmissions,blockSize=64,step_size=None,compressor='gzip'):
        self.submissions= listOfSubmissions
        self.s=blockSize

        self.compress= lzma.compress
        if(compressor!='lzma'):
            self.compress=gzip.compress

        if(step_size is None):
            self.step=self.s/2
        else:
            self.step=step_size
    
    def blockCheck(self,append_ncd=False):
        #FileCache={}
        BlockCache={}
        #CompressedFileCache= {}
        compressedBlockCache={}
        results={}
        #ncd=np.zeros((len(self.submissions),len(self.submissions)))
        for i in range(len(self.submissions)):
            A=self.submissions[i].code
            Z_A=len(self.compress(A.encode()))
            # FileCache[self.submissions[i].getstudentID()]=A
            # CompressedFileCache[self.submissions[i].getstudentID()]=Z_A
            for j in range(len(self.submissions)):
                FNCD=[]
                if(self.submissions[j].getstudentID() in compressedBlockCache):
                    for z_b,bl in zip(compressedBlockCache[self.submissions[j].getstudentID()],BlockCache[self.submissions[j].getstudentID()]):
                        Z_Ab= len(self.compress(A.encode()+bl.encode()))
                        #FNCD.append((Z_Ab-z_b)/Z_A)
                        FNCD.append((Z_Ab-Z_A)/z_b)
                else:
                    compressedBlockCache[self.submissions[j].getstudentID()]=[]
                    BlockCache[self.submissions[j].getstudentID()]=[]
                    for k in  range (0,len(self.submissions[j].code),self.step):
                        b=self.submissions[j].code[k:k+self.s]
                        Z_b=len(self.compress(b.encode()))
                        compressedBlockCache[self.submissions[j].getstudentID()].append(Z_b)
                        BlockCache[self.submissions[j].getstudentID()].append(b)
                        Z_Ab= len(self.compress(A.encode()+b.encode()))
                        #FNCD.append((Z_Ab-Z_b)/Z_A)
                        FNCD.append((Z_Ab-Z_A)/Z_b) # how much information does this block b have that is the same as in A

                # if(ncd[i,j]==0 and i!=j):
                #     Z_AB=len(self.compress(A.encode()+self.submissions[j].code.encode()))
                #     Z_B= len(self.compress(self.submissions[j].code.encode()))
                #     ncd[i,j]=(Z_AB- min(Z_A,Z_B))/max(Z_A,Z_B)
                #     ncd[j,i]=(Z_AB- min(Z_A,Z_B))/max(Z_A,Z_B)
                if(append_ncd):
                    Z_AB=len(self.compress(A.encode()+self.submissions[j].code.encode()))
                    Z_B= len(self.compress(self.submissions[j].code.encode()))          
                    FNCD.append((Z_AB- min(Z_A,Z_B))/max(Z_A,Z_B))
                    
                FNCD.sort()
                key= str(self.submissions[i].getstudentID())+"_"+str(self.submissions[j].getstudentID())
                results[key]=FNCD
                
        return results

    def Check(self):
        ncd=np.zeros((len(self.submissions),len(self.submissions)))
        for i in range(len(self.submissions)):
            A=self.submissions[i].code
            Z_A=len(self.compress(A.encode()))
            # FileCache[self.submissions[i].getstudentID()]=A
            # CompressedFileCache[self.submissions[i].getstudentID()]=Z_A
            for j in range(len(self.submissions)):
                if(ncd[i,j]==0 and i!=j):
                    Z_AB=len(self.compress(A.encode()+self.submissions[j].code.encode()))
                    Z_B= len(self.compress(self.submissions[j].code.encode()))
                    ncd[i,j]=(Z_AB- min(Z_A,Z_B))/max(Z_A,Z_B)
                    ncd[j,i]=(Z_AB- min(Z_A,Z_B))/max(Z_A,Z_B)

        return ncd

class NCDShuffle:
    def __init__(self,listOfSubmissions, block_size = 64, step_size = None, compressor='lzma') -> None:
        self.similarityMatrix = np.zeros((len(listOfSubmissions),len(listOfSubmissions)))
        self.submissions= listOfSubmissions
        self.block_size=block_size
        self.compressor=compressor
        self.lzma_filters = [
            {
            "id": lzma.FILTER_LZMA2, 
            "preset": 9 | lzma.PRESET_EXTREME, 
            "dict_size": 4096, # a big enough dictionary, but not more than needed, saves memory
            "mode": lzma.MODE_NORMAL,
            "mf": lzma.MF_BT4
            }
        ]

        if(step_size is None):
            self.step=self.block_size/2
        else:
            self.step=step_size

    def Z(self,x):
        if(self.compressor!='lzma'):
            return len(gzip.compress(x))
        return len(lzma.compress(x,filters=self.lzma_filters))
    
    def ncd_shuffle(self, adaptive_windows=False):
        # with Pool(multiprocessing.cpu_count()) as p:
        #     compressed_lengths, blocks = self.precomputeCompress()
        #     print("Pre-processing done!")
        #     for i, submission in enumerate(self.submissions):
        #         p.apply_async(self.calculate_row, args = (submission.getstudentID(),i,compressed_lengths, blocks), callback = self.populate_matrix) 
        #     p.close()
        #     p.join()
        compressed_lengths, blocks = self.adaptivePrecomputeCompress() if adaptive_windows else self.precomputeCompress()
        print("Pre-processing done!")
        for i, submission in enumerate(self.submissions):
            info=self.calculate_row(submission.getstudentID(),i,compressed_lengths, blocks)
            self.populate_matrix(info)
        
        return self.similarityMatrix

    def calculate_similarity(self,a, b, compressed_A, compressed_B):
        similarities = []
        zb= list(zip(b,compressed_B))

        for block_A, Z_A in zip(a,compressed_A):
            min_NCD =1.0
            min_index=-1
            for j,[block_B,Z_B] in enumerate(zb):
                ncd = (self.Z(block_A+block_B) - min(Z_A,Z_B))/max(Z_A,Z_B)
                if(ncd<min_NCD):
                    min_NCD = ncd
                    min_index = j
            
            if(min_index!=-1):
                zb.pop((min_index))
                similarities.append(min_NCD)
            
            return sum(similarities)/len(similarities)
        
    def calculate_row(self,studentID,row, compressed_lengths, blocks):
        result =[0.0 for i in range(len(self.submissions))]
        print(f"calculating row: {row}\n",flush= True)
        for i, submission in enumerate(self.submissions[row+1:],row+1):
                result[i] = self.calculate_similarity(blocks[studentID],
                    blocks[submission.getstudentID()],
                    compressed_lengths[studentID],
                    compressed_lengths[submission.getstudentID()])
        return result,row

    def populate_matrix(self, info):
        result, row = info
        #print(f"setting row {row} with result {result}\n", flush= True)
        self.similarityMatrix[row]= result

    def precomputeCompress(self):
        compressed_lengths = {}
        blocks={}
        for submission in self.submissions:
            blocks[submission.getstudentID()]= [submission.code[k:k+self.block_size].encode() for k in  range (0,len(submission.code),self.step)]
            compressed_lengths[submission.getstudentID()] = list(map(self.Z, blocks[submission.getstudentID()]))
        
        #print(compressed_lengths)

        return compressed_lengths,blocks
    
    def adaptivePrecomputeCompress(self):
        compressed_lengths = {}
        blocks={}
        for submission in self.submissions:
            if(len(submission.code)<self.block_size):
                new_block_size = int(len(submission.code)/4)
                new_step_size = int(new_block_size/2)
                blocks[submission.getstudentID()]= [submission.code[k:k+new_block_size].encode() for k in  range (0,len(submission.code),new_step_size)]

            blocks[submission.getstudentID()]= [submission.code[k:k+self.block_size].encode() for k in  range (0,len(submission.code),self.step)]
            compressed_lengths[submission.getstudentID()] = list(map(self.Z, blocks[submission.getstudentID()]))
        
        #print(compressed_lengths)

        return compressed_lengths,blocks
        
