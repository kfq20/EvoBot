import torch
from torch import nn
from torch_geometric.nn import RGCNConv,FastRGCNConv,GCNConv,GATConv
import torch.nn.functional as F
from trl import BasePairwiseJudge
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc

class BotRGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, embedding_dimension=128, dropout=0.3):
        super(BotRGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        
        # RGCNConv layer for relational graph convolution, num_relations=2 indicates two types of relations in the graph
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(int(embedding_dimension/2),2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        assert des.shape[0] == tweet.shape[0] == num_prop.shape[0] == cat_prop.shape[0], "All input tensors must have the same batch size"
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class BotGAT(nn.Module):
    def __init__(self, hidden_dim, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, dropout=0.3):
        super(BotGAT, self).__init__()
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dim, 2)

        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type=None):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        x = self.dropout(x)
        x = self.linear_relu_input(x)
        x = self.gat1(x, edge_index)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x

def accuracy(output, labels):
            preds = output.max(1)[1].type_as(labels)
            correct = preds.eq(labels).double()
            correct = correct.sum()
            return correct / len(labels)

def train_discrim(model, loss_func, optimizer, epochs, inputs):
    acc_train, loss_train = 0, 0
    for epoch in range(epochs):
        model.train()
        total_acc_train, total_acc_val, total_loss = 0, 0, 0
        for input in inputs.values():
            des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx = input[0],input[1],input[2],input[3],input[4],input[5],input[6],input[7],input[8]
            output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
            loss_train = loss_func(output[train_idx], labels[train_idx])
            acc_train = accuracy(output[train_idx], labels[train_idx])
            acc_val = accuracy(output[val_idx], labels[val_idx])
            total_acc_train += acc_train.item()
            total_acc_val += acc_val.item()
            total_loss += loss_train

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(total_loss.item()/len(inputs)),
                'acc_train: {:.4f}'.format(total_acc_train/len(inputs)),
                'acc_val: {:.4f}'.format(total_acc_val/len(inputs)),)
    return acc_train,loss_train


def test_discrim(model, loss_func, inputs):
    avg_acc = 0
    avg_f1 = 0
    model.eval()
    for name, input in inputs.items():
        des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx = input[0],input[1],input[2],input[3],input[4],input[5],input[6],input[7],input[8], input[9]
        output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
        loss_test = loss_func(output[test_idx], labels[test_idx])
        acc_test = accuracy(output[test_idx], labels[test_idx])
        output=output.max(1)[1].to('cpu').detach().numpy()
        label=labels.to('cpu').detach().numpy()
        f1=f1_score(label[test_idx],output[test_idx])
        precision=precision_score(label[test_idx],output[test_idx])
        recall=recall_score(label[test_idx],output[test_idx])
        fpr, tpr, thresholds = roc_curve(label[test_idx], output[test_idx], pos_label=1)
        Auc=auc(fpr, tpr)
        print(f"Test set results on {name}:",
                "test_loss= {:.4f}".format(loss_test.item()),
                "test_accuracy= {:.4f}".format(acc_test.item()),
                "precision= {:.4f}".format(precision.item()),
                "recall= {:.4f}".format(recall.item()),
                "f1_score= {:.4f}".format(f1.item()),
                #"mcc= {:.4f}".format(mcc.item()),
                "auc= {:.4f}".format(Auc.item()),
                )
        avg_acc += acc_test.item()
        avg_f1 += f1.item()
    return avg_acc / len(inputs), avg_f1 / len(inputs)