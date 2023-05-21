from NCD import NCD
import numpy as np
#from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import scipy.cluster.hierarchy as hc
import scipy.spatial.distance as dist
import utils
from BlockNCD import BlockNCD,NCDShuffle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class NCDClusterer:

    def __init__(self,arr=None, Studentsubmissions=None):
        """
        matrix is an array of NCD objects
        use one of the utils functions to create 
        """
        self.submissions=Studentsubmissions
        self.matrixOfNCD=arr
        self.dimension=len(self.submissions)
        self.similarityMatix=np.zeros((self.dimension,self.dimension))


    def checkBlockSimilarity(self,k=15,b=64,s=32,append_ncd=False):
        blockNCD=BlockNCD(self.submissions,b,s)
        results=blockNCD.blockCheck(append_ncd=append_ncd)
        #print(self.similarityMatix2)
        for i in range(len(self.submissions)):
            for j in range(i+1,len(self.submissions)):
                 key1=str(self.submissions[i].getstudentID())+"_"+str(self.submissions[j].getstudentID())
                 key2=str(self.submissions[j].getstudentID())+"_"+str(self.submissions[i].getstudentID())
                 value=results[key1]+results[key2]
                 value.sort() #replace with merge of 2 ordered list algo
                 total=0
                 for m in range(min(len(value),k)):
                     total+=value[m]
                 self.similarityMatix[i,j]=total/min(len(value),k)
                 self.similarityMatix[j,i]=total/min(len(value),k)
        
        return self.similarityMatix
    
    def checkShuffleSimilarity(self,b=64,s=32, adaptive_windows=False):
        self.similarityMatix = NCDShuffle(self.submissions, b, s).ncd_shuffle(adaptive_windows)
        for i in range(self.similarityMatix.shape[0]):
            for j in range(i+1,self.similarityMatix.shape[1]):
                self.similarityMatix[j,i]=self.similarityMatix[i,j]
        return self.similarityMatix

                
    def checkSimilarity(self):
        # obj= NCD("","")
        # self.similarityMatix=obj.batchCheck(self.submissions)
        blockNCD=BlockNCD(self.submissions,64,32)
        self.similarityMatix=blockNCD.Check()
        return self.similarityMatix

    def cluster(self, clusteringLikage='ward', sigmoid=-1, normalize=False):
        """
        clustering Linkage:
        ‘average’ uses the average of the distances of each observation of the two sets.

        ‘complete’ or ‘maximum’ linkage uses the maximum distances between all observations of the two sets.

        ‘single’ uses the minimum of the distances between all observations of the two sets.

        """
        # clusters=AgglomerativeClustering(linkage='average',affinity='precomputed',n_clusters=nclusts).fit(self.similarityMatix)
        # labels=clusters.labels_
        mat =self.similarityMatix
        if(sigmoid>=0):
            mat= utils.sigmoid(mat, mean=sigmoid)
            for i in range(mat.shape[0]):
                mat[i,i]=0
        if(normalize):
            mat = (mat - np.min(mat))/(np.max(mat)-np.min(mat))
        condensed_dist_vector= dist.squareform(mat)

        clusters= hc.linkage(condensed_dist_vector, method=clusteringLikage,optimal_ordering=True)
        
        return clusters

class Clusterer:
    def __init__(self,simMatrix):
        self.simMatrix=simMatrix

    def cluster(self, clusteringLikage='ward',sigmoid=False):
        """
        clustering Linkage:
        ‘average’ uses the average of the distances of each observation of the two sets.

        ‘complete’ or ‘maximum’ linkage uses the maximum distances between all observations of the two sets.

        ‘single’ uses the minimum of the distances between all observations of the two sets.

         """
        # clusters=AgglomerativeClustering(linkage='average',affinity='precomputed',n_clusters=nclusts).fit(self.similarityMatix)
        # labels=clusters.labels_
        if(sigmoid):
            mat= utils.sigmoid(self.simMatrix)
            for i in range(mat.shape[0]):
                mat[i,i]=0
        else:
            mat =self.simMatrix
        condensed_dist_vector= dist.squareform(mat)
        clusters= hc.linkage(condensed_dist_vector, method=clusteringLikage, optimal_ordering=True)

        return clusters
    
    def kmeans(self,vecs,k=7):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(vecs)
        label=kmeans.labels_ 
        u_labels = np.unique(label)
 
        #plotting the results:
 
        for i in u_labels:
            plt.scatter(vecs[label == i , 0] , vecs[label == i , 1] , label = i)
        plt.legend()
        plt.show()

        return  label
    
    def dbscan(self, eps=0.1, min_samples=7):
        return DBSCAN(eps=eps, min_samples=min_samples).fit(self.simMatrix)
