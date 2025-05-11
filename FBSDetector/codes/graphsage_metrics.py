import sys
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from graph_models import GraphDataProcessor, GraphSAGEModel, GNNTrainer

def calculate_class_metrics(y_true, y_pred, num_classes):
    conf_matrix = confusion_matrix(y_true, y_pred)
    metrics = []
    
    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        TN = conf_matrix.sum() - (TP + FP + FN)
        
        metrics.append({
            'Class': i,
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN),
        })
    
    return metrics

def main():
    # File paths
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
    
    # Initialize and train GraphSAGE model
    num_classes = len(torch.unique(data_obj.y))
    model = GraphSAGEModel(data_obj.x.shape[1], num_classes)
    trainer = GNNTrainer(model, data_obj, epochs=200)
    trainer.train()
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        pred = model(data_obj.x, data_obj.edge_index).argmax(dim=1)
        y_true = data_obj.y[data_obj.test_mask].cpu().numpy()
        y_pred = pred[data_obj.test_mask].cpu().numpy()
    
    # Calculate metrics
    metrics = calculate_class_metrics(y_true, y_pred, num_classes)
    
    # Create DataFrame for tabular report
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df[['Class', 'TP', 'TN', 'FP', 'FN']]
    
    # Save results
    print("GraphSAGE Model Metrics\n")
    print("=======================\n\n")
    # Add tabular summary
    print("Summary Table:\n")
    print("-" * 80 + "\n")
    print(metrics_df.to_string(index=False, float_format=lambda x: '{:.3f}'.format(x)))
    print("\n\n")

if __name__ == "__main__":
    main()
