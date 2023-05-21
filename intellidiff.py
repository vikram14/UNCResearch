import torch
import numpy
import re
import nltk
from sentence_transformers import SentenceTransformer,util


#nltk.download('punkt')
class IntelliDiff:
    def __init__(self,modelName='all-MiniLM-L6-v2') -> None:
        
        self.model = SentenceTransformer(modelName)
    
    def getSentences(self, explanation):
        idx=explanation.find("*/")
        if(idx>=0):
            expl=explanation[0:idx]
        else:
            expl=explanation
        expl=re.sub(r'/|\*|\n|\r','',expl)
        expl= expl.replace('.',' . ')
        sentences= nltk.sent_tokenize(expl)
        avg_len=sum(list(map( lambda x: len(x), sentences[0:len(sentences)-1])))/len(sentences)
        if(len(sentences[-1])< avg_len-10 ): # if last sentence is an incomplete description
            sentences = sentences[0:len(sentences)-1]
        
        return sentences
    
    def getSentenceEmbeddings(self, sentences):
        return list(map(lambda x : (self.model.encode(x)).reshape((1,-1)),sentences))
    
    def diff(self, expl_a, expl_b, threshold=0.6):
        """returns all sentences in b and not in a. b represents target sample"""
        sent_a=self.getSentences(expl_a)
        sent_b=self.getSentences(expl_b)
        a=self.getSentenceEmbeddings(sent_a)
        b= self.getSentenceEmbeddings(sent_b)
        result=set()
        for i,embedding_b in enumerate(b):
            flag=False
            for j,embedding_a in enumerate(a):
                sim=util.cos_sim(embedding_a, embedding_b)[0,0]
                #print(f"B : {sent_b[i]} \n A : {sent_a[j]}; \n sim_score:{sim};")
                if(sim>=threshold):
                    flag=True
                    break
            if(not flag):
                result.add(sent_b[i])
        return list(result)

            

        
        
