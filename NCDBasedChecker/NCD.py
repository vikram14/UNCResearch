import lzma
import numpy as np

class NCD:

    def __init__(self,x,y):
        """
        x,y are binary strings
        """
        self.file1=x
        self.file2=y
    
    def setFile1(self,x):
        self.file1=x
    
    def getFile1(self):
        return self.file1
    
    def setFile2(self,y):
        self.file2=y
    
    def getFile2(self):
        return self.getFile2

    @classmethod
    def fromFileName(cls,x,y):
        """
        reads file from file name and returns instance of this class
        """
        return cls(open(x,'rb').read(),open(y,'rb').read())

    def similarityCheck(self):
        Z_x=len(lzma.compress(self.file1))
        Z_y=len(lzma.compress(self.file2))
        Z_xy=len(lzma.compress(self.file1+self.file2))
        print(f'z_x={Z_x}')
        print(f'z_y={Z_y}')
        print(f'z_xy={Z_xy}')
        ncd= (Z_xy-min(Z_x,Z_y))/max(Z_x,Z_y)
        return ncd
    
    def batchCheck(self, arr):
        ncd=np.zeros((len(arr),len(arr)))
        y_cache={}
        for i,s_i in enumerate(arr):
            if arr[i].getstudentID() not in y_cache:
                C_x=lzma.compress(s_i.code.encode())
                y_cache[arr[i].getstudentID()]=len(C_x)

            Z_x=y_cache[arr[i].getstudentID()]

            for j in range(i+1, len(arr)):
                if arr[j].getstudentID() not in y_cache:
                    C_y=lzma.compress(arr[j].code.encode())
                    y_cache[arr[j].getstudentID()]=len(C_y)
                
                Z_y= y_cache[arr[j].getstudentID()]
                Z_xy=len(lzma.compress(s_i.code.encode()+arr[j].code.encode()))

                ncd[i,j] = (Z_xy-min(Z_x,Z_y))/max(Z_x,Z_y)
                ncd[j,i] = (Z_xy-min(Z_x,Z_y))/max(Z_x,Z_y)
        return ncd
           
