from vectorize import load_json, save_json,vectorize
from sklearn.metrics.cluster import adjusted_rand_score,rand_score,normalized_mutual_info_score,pair_confusion_matrix,fowlkes_mallows_score
from sklearn.metrics import ConfusionMatrixDisplay

def clusters_members(clust,key='label'):
    labelToStudents={}
    for k, data in clust.items():
        if(data[key] not in labelToStudents):
            labelToStudents[data[key]]=[]
        labelToStudents[data[key]].append(k)
    return labelToStudents

def iou(clust1,clust2):
    overlap={}
    precision={}
    for k,v in clust1.items():
        set_v =set(v)
        max_key=None
        max_iou=0
        accuracy = 0
        for k1,v1 in clust2.items():
            intersection = set_v.intersection(v1)
            union = set_v.union(v1)
            if(len(intersection)/len(union)>=max_iou):
                max_iou =len(intersection)/len(union)
                max_key = (k,k1)
                accuracy = len(intersection)/len(v1)

        overlap[max_key]= max_iou 
        precision[max_key]= accuracy

    print(f'iou:{overlap}')
    print("\n")
    print(f'accuracy:{precision}')


def compare(file_path_pred, file_path_manual,key_manual ='label_order1'):
    manual_clust= load_json(file_path_manual)
    pred_clust =load_json(file_path_pred)
    clusters_members_manual=clusters_members(manual_clust,key=key_manual)
    clusters_members_pred=clusters_members(pred_clust)
    pred_clustering =[v['label'] for k, v in pred_clust.items()]
    manual_clustering =[v[key_manual] for k, v in manual_clust.items()]
    cm= pair_confusion_matrix(manual_clustering,pred_clustering)

    print('*********************************')
    print()
    iou(clusters_members_pred,clusters_members_manual)
    print()
    print(f'rand score: {rand_score(pred_clustering, manual_clustering)}')
    print(f'adjusted rand score: {adjusted_rand_score(pred_clustering, manual_clustering)}')
    print(f'normalized mutual information: {normalized_mutual_info_score(pred_clustering, manual_clustering)}')
    print(f'fowlkes mallows score: {fowlkes_mallows_score(manual_clustering,pred_clustering)}')
    print('''confusion matrix:
    (0,0) - True Negatives. Pair is not clustered together in the predicted and manual clustering.
    (0,1) - False Positives. Pair is clustered together in prediction but not manual clustering.
    (1,0) - False Negatives. Pair is not clustered together in prediction but is clustered together in manual clustering.
    (1,1) - True Positives.  Pair is clustered together in the predicted and manual clustering.''')
    print()
    return ConfusionMatrixDisplay(confusion_matrix=cm)
