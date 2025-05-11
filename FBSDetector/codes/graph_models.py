import sys
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv, TransformerConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class GraphDataProcessor:
    def __init__(self, filepath, dataset_type="nas"):
        self.data = pd.read_csv(filepath)
        self.dataset_type = dataset_type
        self.source_col = 'nas-eps_nas_msg_emm_type_value' if dataset_type == "nas" else 'lte-rrc_c1_showname'
        
        # Remap labels to be zero-based consecutive integers
        unique_labels = sorted(self.data['label'].unique())
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        print(f"Number of unique classes: {self.num_classes}")
        print(f"Label mapping: {self.label_mapping}")
        
        # Create node mapping first
        unique_values = self.data[self.source_col].unique()
        self.node_mapping = {node: idx for idx, node in enumerate(unique_values)}
        self.num_nodes = len(unique_values)
        
        # Create edge indices and features
        self.edge_index, self.features, self.labels = self._process_data()

    def _process_data(self):
        # Create edge indices with bounds checking
        edge_indices = []
        max_idx = len(self.node_mapping) - 1
        
        for i in range(len(self.data) - 1):
            try:
                source = self.node_mapping[self.data.iloc[i][self.source_col]]
                target = self.node_mapping[self.data.iloc[i + 1][self.source_col]]
                
                # Verify indices are within bounds
                if 0 <= source <= max_idx and 0 <= target <= max_idx:
                    edge_indices.append([source, target])
                else:
                    print(f"Skipping edge due to out of bounds indices: {source} -> {target}")
            except KeyError as e:
                print(f"Warning: Node not found in mapping: {e}")
                continue
        
        if not edge_indices:
            print("Warning: No valid edges found!")
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
            print(f"Created edge_index with shape: {edge_index.shape}")
            print(f"Max index in edge_index: {edge_index.max().item()}")
            print(f"Number of nodes: {self.num_nodes}")
        
        # Ensure features exclude the source column
        feature_cols = [col for col in self.data.columns if col not in ['label', self.source_col]]
        features = torch.FloatTensor(self.data[feature_cols].values)
        
        # Map labels using the label_mapping
        labels = torch.LongTensor([self.label_mapping[label] for label in self.data['label']])
        
        return edge_index, features, labels

    def get_data_object(self):
        train_idx, test_idx = train_test_split(np.arange(len(self.data)), test_size=0.2, random_state=42)
        train_mask = torch.BoolTensor(np.isin(np.arange(len(self.data)), train_idx))
        test_mask = ~train_mask
        
        data_obj = Data(
            x=self.features,
            edge_index=self.edge_index,
            y=self.labels,
            train_mask=train_mask,
            test_mask=test_mask
        )
        return data_obj

class BaseGNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(BaseGNN, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

    def forward(self, x, edge_index):
        raise NotImplementedError("Subclasses should implement forward method.")

class GATModel(BaseGNN):
    def __init__(self, num_features, num_classes, num_heads=2):
        super(GATModel, self).__init__(num_features, num_classes)
        self.conv1 = GATConv(num_features, num_classes, heads=num_heads)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

class GATv2Model(BaseGNN):
    def __init__(self, num_features, num_classes, num_heads=2):
        super(GATv2Model, self).__init__(num_features, num_classes)
        self.conv1 = GATv2Conv(num_features, num_classes, heads=num_heads)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

class GCNModel(BaseGNN):
    def __init__(self, num_features, num_classes):
        super(GCNModel, self).__init__(num_features, num_classes)
        self.conv1 = GCNConv(num_features, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGEModel(BaseGNN):
    def __init__(self, num_features, num_classes):
        super(GraphSAGEModel, self).__init__(num_features, num_classes)
        self.conv1 = SAGEConv(num_features, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphTransformerModel(BaseGNN):
    def __init__(self, num_features, num_classes):
        super(GraphTransformerModel, self).__init__(num_features, num_classes)
        self.conv1 = TransformerConv(num_features, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)

class GNNTrainer:
    def __init__(self, model, data, epochs=100, lr=0.01):
        self.model = model
        self.data = data
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
        
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.data.x, self.data.edge_index).argmax(dim=1)
            y_true = self.data.y[self.data.test_mask].cpu().numpy()
            y_pred = pred[self.data.test_mask].cpu().numpy()
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            accuracy = accuracy_score(y_true, y_pred)
            
            results = [precision, recall, f1, accuracy]
            return results

if __name__ == "__main__":
    # select the dataset (uncomment to use)
    # takke the datset from the command line
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <[msa_nas/msa_rrc.csv]>")
        sys.exit(1)
    dataset = sys.argv[1]
    dataset_type = ""
    if "nas" in dataset:
        dataset_type = "nas"
    elif "rrc" in dataset:
        dataset_type = "rrc"
    else:
        raise ValueError("Invalid dataset type. Use 'nas' or 'rrc'.")    
    processor = GraphDataProcessor(dataset, dataset_type=dataset_type)

    data_obj = processor.get_data_object()
    
    # Update models dictionary to use correct number of classes
    models = {
        "GAT": GATModel(data_obj.x.shape[1], processor.num_classes),
        "GATv2": GATv2Model(data_obj.x.shape[1], processor.num_classes),
        "GCN": GCNModel(data_obj.x.shape[1], processor.num_classes),
        "GraphTransformer": GraphTransformerModel(data_obj.x.shape[1], processor.num_classes),
        "GraphSAGE": GraphSAGEModel(data_obj.x.shape[1], processor.num_classes)
    }
    final_results = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'F1-Score', 'Accuracy',])
    for name, model in models.items():
        trainer = GNNTrainer(model, data_obj)
        trainer.train()
        model_results = trainer.evaluate()
        final_results = pd.concat([final_results, pd.DataFrame([{"Model": name, "Precision": model_results[0], 
                                                "Recall": model_results[1], "F1-Score": model_results[2], 
                                                "Accuracy": model_results[3]}])],ignore_index=True)
    print("\nFinal Results:")
    print(final_results)
