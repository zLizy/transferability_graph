#https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py

import sys
import torch
from graph import Graph

class Classifier():
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_0 = x[edge_index[0]]
        edge_feat_1= x[edge_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_0 * edge_feat_1).sum(dim=-1)

class N2VModel():
    def __init__(
        self,
        edge_index,
        embedding_dim=128,
        walk_length=5,
        context_size=5,
        walks_per_node=5,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=True,
    ):
        self.base = Node2Vec(
            data.edge_index,
            embedding_dim=embedding_dim,
            walk_length= walk_length, #20,
            context_size=context_size,
            walks_per_node=walks_per_node, #10,
            num_negative_samples=num_negative_samples,
            p=1,
            q=1,
            sparse=True,
        ).to(device)
        self.classifier = Classifier()
    
    def forward(self,data):
        pred = self.classifier(
            x,
            edge_index,
        )
        return pred
    
    @torch.no_grad()
    def test(self):
        self.base.eval()
        z = self.base()
        # acc = model.test(z[data.train_mask], data.y[data.train_mask],
        #                  z[data.test_mask], data.y[data.test_mask],
        #                  max_iter=150)
        return acc
    
    def train(self):
        num_workers = 0 if sys.platform.startswith('win') else 4
        loader = self.base.loader(batch_size=128, shuffle=True,
                            num_workers=num_workers)
        optimizer = torch.optim.SparseAdam(list(self.base.parameters()), lr=0.01)

        for epoch in range(1, 11):
            self.base.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = self.base.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            loss = total_loss / len(loader)
            # acc = test()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}') #, Acc: {acc:.4f}')
            

def node2vec_train(args,df_perf,data_dict,evaluation_dict,setting_dict,batch_size):

    ## Construct a graph
    graph = Graph(
        edge_index_model_to_dataset,
        edge_index_dataset_to_dataset,
        )
    data = graph.data

    model = N2VModel(data.edge_index)
    model.train()


    

    