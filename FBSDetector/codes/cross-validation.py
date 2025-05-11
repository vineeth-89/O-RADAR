import sys
import torch
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import KFold, LeaveOneGroupOut
from typing import List, Tuple

class GraphDataPreprocessor:
    def __init__(self, data_path: str, dataset_type: str):
        self.dataset_type = dataset_type
        try:
            self.data = pd.read_csv(data_path)
            self._validate_data()
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
            
        self.graph = None
        self.adj_matrix = None
        self.features = None
        self.labels = None
        self.unique_values = None
        
    def _validate_data(self) -> None:
        """Validate required columns exist in the dataset."""
        if self.dataset_type == 'nas':
            required_columns = ['nas-eps_nas_msg_emm_type_value', 'label']
        elif self.dataset_type == 'rrc':
            required_columns = ['lte-rrc_c1_showname', 'label']
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
    def create_graph(self) -> None:
        """Create a directed graph from the sequential data."""
        self.graph = nx.DiGraph()
        if self.dataset_type == 'nas':
            self.unique_values = self.data['nas-eps_nas_msg_emm_type_value'].unique()
            source_col = 'nas-eps_nas_msg_emm_type_value'
            target_col = 'nas-eps_nas_msg_emm_type_value'
        elif self.dataset_type == 'rrc':
            self.unique_values = self.data['lte-rrc_c1_showname'].unique()
            source_col = 'lte-rrc_c1_showname'
            target_col = 'lte-rrc_c1_showname'
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        self.graph.add_nodes_from(self.unique_values)
        
        for i in range(len(self.data) - 1):
            source = self.data.loc[i, source_col]
            target = self.data.loc[i + 1, target_col]
            edge_label = self.data.loc[i + 1, 'label']
            self.graph.add_edge(source, target, label=edge_label)
    
    def prepare_features(self) -> None:
        """Prepare feature matrix and labels."""
        # Convert all columns to numeric
        for col in self.data.columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        self.data = self.data.fillna(0)
        self.features = torch.FloatTensor(self.data.drop('label', axis=1).values)
        self.labels = torch.LongTensor(self.data['label'].values)
        
        # Create adjacency matrix
        self.adj_matrix = nx.adjacency_matrix(self.graph)
        self.adj_matrix = torch.FloatTensor(self.adj_matrix.toarray())

class GraphSAGEModel(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(num_features, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return F.log_softmax(x, dim=1)

class CrossValidator:
    def __init__(self, model: nn.Module, num_epochs: int = 100, lr: float = 0.01,
                 patience: int = 10):
        self.model = model
        self.num_epochs = num_epochs
        self.lr = lr
        self.patience = patience
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5)
        
    def train_model(self, data: Data) -> Tuple[float, int]:
        """Train model with early stopping."""
        best_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(self.num_epochs):
            loss = self.train_epoch(data)
            
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        return best_loss, best_epoch
        
    def train_epoch(self, data: Data) -> float:
        """Train for one epoch and return loss."""
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(data.x, data.edge_index)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def evaluate(self, data: Data, test_data: pd.DataFrame) -> Tuple[torch.Tensor, float]:
        """Evaluate model and return predictions and accuracy."""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(data.x, data.edge_index).argmax(dim=1)
            correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
            total = data.test_mask.sum().item()
            accuracy = correct / total
            return pred, accuracy
    
    def print_results(self, test_labels: np.ndarray, test_data: pd.DataFrame, 
                     predictions: torch.Tensor, test_mask: torch.Tensor) -> None:
        """Print detailed results of the evaluation."""
        test_indices = torch.arange(len(test_data))[test_mask]
        results_df = test_data.iloc[test_indices].copy()
        results_df['predicted_label'] = predictions[test_mask].numpy()
        
        print(f"Test Fold: {test_labels}")
        print(results_df[['label', 'predicted_label']])
        print("="*30)
        
        counts = results_df.groupby(['label', 'predicted_label']).size().unstack(fill_value=0)
        print("Class Counts:")
        print(counts)
        print("="*30)

def main():
    try:
        # Initialize data preprocessor
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
        preprocessor = GraphDataPreprocessor(dataset, dataset_type=dataset_type)

        preprocessor.create_graph()
        preprocessor.prepare_features()
        
        # Model parameters
        num_features = preprocessor.features.shape[1]
        num_classes = preprocessor.labels.max().item() + 1  # Ensure num_classes is set correctly
        hidden_dim = 64
        
        # Initialize leave-one-group-out cross-validation
        logo = LeaveOneGroupOut()
        accuracies = []
        results = []
        all_predictions = []
        
        # Perform leave-one-class-out cross-validation
        for fold, (train_idx, test_idx) in enumerate(logo.split(preprocessor.data, groups=preprocessor.data['label'])):
            print(f"\nFold {fold + 1}/{num_classes}")
            
            # Prepare masks
            train_mask = torch.zeros(len(preprocessor.data), dtype=torch.bool)
            test_mask = torch.zeros(len(preprocessor.data), dtype=torch.bool)
            train_mask[train_idx] = True
            test_mask[test_idx] = True
            
            # Initialize model and cross-validator
            model = GraphSAGEModel(num_features, num_classes, hidden_dim)
            validator = CrossValidator(model)
            
            # Create PyTorch Geometric data object
            data_obj = Data(
                x=preprocessor.features,
                edge_index=preprocessor.adj_matrix.nonzero().T,
                y=preprocessor.labels
            )
            data_obj.train_mask = train_mask
            data_obj.test_mask = test_mask
            
            # Training
            best_loss, best_epoch = validator.train_model(data_obj)
            
            # Evaluation
            predictions, accuracy = validator.evaluate(data_obj, preprocessor.data)
            accuracies.append(accuracy)
            
            print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}")
            results.append({
                'Fold': fold + 1,
                'Test Class': preprocessor.data['label'].iloc[test_idx[0]],
                'Accuracy': accuracy
            })
            all_predictions.append({
                'True Label': preprocessor.data['label'].iloc[test_idx].values,
                'Predicted Label': predictions[test_mask].numpy()
            })
            validator.print_results(preprocessor.data['label'].iloc[test_idx].values, preprocessor.data, predictions, test_mask)
        
        # Summarize all predictions in a DataFrame
        all_predictions_df = pd.DataFrame({
            'True Label': np.concatenate([pred['True Label'] for pred in all_predictions]),
            'Predicted Label': np.concatenate([pred['Predicted Label'] for pred in all_predictions])
        })
        # print("\nAll Predictions:")
        # print(all_predictions_df)
        
        # Create a pivot table
        pivot_table = all_predictions_df.pivot_table(index='True Label', columns='Predicted Label', aggfunc='size', fill_value=0)
        pivot_table = pivot_table.reindex(columns=pivot_table.index, fill_value=0)
        print("\nPivot Table:")
        print(pivot_table)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()