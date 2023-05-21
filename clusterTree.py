from queue import Queue
from typing import Any
import numpy as np
import scipy.cluster.hierarchy as hc

class ClusterTree():
    def __init__(self,linkage:np.ndarray,IndexToStudentId:dict) -> None:
        self.cluster:dict={}
        self.linkage:np.ndarray = linkage
        self.IndexToStudentIdMap:dict= IndexToStudentId
        self.StudentIdtoIndexMap:dict = {v:k for k,v in self.IndexToStudentIdMap.items()}
        self.n:int = self.linkage.shape[0]+1
        self.cluster:dict={i:{'id':i,'leaf':True,'size':1, 'parent':None, 'children':None,'distance':0} for i in range(self.n)}
        self.root = 2*self.n-2

    def build_tree(self)->None:
        for i in range(0, self.n-1):
            c1,c2,distance,size = self.linkage[i]
            self.cluster[i+self.n] = {'id':i+self.n,'leaf':False, 'size': size, 'distance':distance, 'children':sorted([c1,c2],reverse=True), 'parent':None}
            self.cluster[c1]['parent'] = i+self.n
            self.cluster[c2]['parent'] = i+self.n
        return self.cluster

    def getSubCluster(self,studentId:str, size)->int:
        index= self.StudentIdtoIndexMap[studentId]
        node=self.cluster[index]
        while(node['size']!=size):
            node=self.cluster[node['parent']]
        return node['id']
  
    def getSubClusterByIndex(self,index)->Any:
        node_id=2*self.n-1-index
        if not (node_id>=self.n and node_id<=self.root):
            print("Invalid index received.")
            return None
        #index =1 highest level merge point, index= 2 2nd highest merge point ....
        return node_id

    def getNSubclustersOf(self,node_id,n): 
        oGClusterIdToNewClusterIdMap={}
        merges=[]
        leaves=[]
        queue = Queue(maxsize=self.n)
        queue.put(node_id)
        id=2*self.cluster[node_id]['size'] - 2
        while(not queue.empty()):
            node = self.cluster[queue.get()]
            if not node['leaf']:
                children=node['children']
                merges.append([children[1],children[0],node['distance'],node['size']])
                oGClusterIdToNewClusterIdMap[node['id']]=id
                id-=1
                for child in children:
                    queue.put(child)
            else:
                leaves.append(node['id'])

        oGIndexToNewIndexMap={v:i for i,v in enumerate(leaves)}

        def transform(x:list):
            r=[x[0],x[1],x[2],x[3]]
            if(x[0] in oGIndexToNewIndexMap):
                r[0]=oGIndexToNewIndexMap[int(x[0])]
            else:
                r[0]=oGClusterIdToNewClusterIdMap[int(x[0])]

            if(x[1] in oGIndexToNewIndexMap):
                r[1]=oGIndexToNewIndexMap[int(x[1])]
            else:
                r[1]=oGClusterIdToNewClusterIdMap[int(x[1])]
            
            return r

        newLinkage= np.asarray(list(map(transform, reversed(merges))))
        print(newLinkage.shape)
        labels = hc.fcluster(newLinkage, t=n, criterion='maxclust')

        og_labels=[0 for i in range(self.n)]

        for i,label in enumerate(labels):
            og_labels[leaves[i]]=label

        return og_labels
        
        
        
            

            

