from numpy import load, product
import annoy
import torch

def create_index(embedding_size,num_trees=20,):
    index=annoy.AnnoyIndex(embedding_size,'angular')
    vectors=(torch.load('./410_descr_vectors.pt',map_location=torch.device('cpu')))
    embeddings=vectors.numpy().tolist()
    for id,vec in enumerate(embeddings):
        index.add_item(int(id),vec)
    index.build(num_trees)
    index.save('410_embeddings_index.ann')

# def getEmbeddings(batch,dataset='test'):
#     if not os.path.isdir('./embeddings'):
#         os.mkdir('./embeddings')
#     encoder=torch.load( r'C:\Users\vikram14\Desktop\Research\CodeTransformer\code-transformer\eval\\codeTransformer.pt')
#     encoder.eval()
#     embeddings=[]
#     image_ids=[]
#     with torch.no_grad():
#         embeddings.append(encoder.forward_batch(batch, need_all_embeddings=True))

# def load_split():
#     with open('./train_split/product_id.pickle','rb') as f:
#         product_id = pickle.load(f)

#     with open('./train_split/partitions.pickle','rb') as f:
#         partition = pickle.load(f)
    
#     with open('./train_split/labels.pickle','rb') as f:
#         labels = pickle.load(f)
    
#     with open('./train_split/image_id.pickle','rb') as f:
#         image_id = pickle.load(f)
    
#     return partition,labels,product_id,image_id

# def getDataLoaders(bs=128,shuffle=True,nw=0):
#     split=load_split()
#     transform=Compose([ToTensor(),Resize((224,224))])
#     train_data=ImageDataset(split[0]['train'],split[1]['train'],split[2]['train'],transform)
#     val_data=ImageDataset(split[0]['val'],split[1]['val'],split[2]['val'],transform)
#     test_data=ImageDataset(split[0]['test'],split[1]['test'],split[2]['test'],transform)
#     TrainLoader=DataLoader(train_data,batch_size=bs,shuffle=shuffle,num_workers=nw)
#     TestLoader=DataLoader(test_data,batch_size=bs,shuffle=shuffle,num_workers=nw)
#     ValLoader=DataLoader(val_data,batch_size=bs,shuffle=shuffle,num_workers=nw)

#     return TrainLoader,TestLoader,ValLoader
