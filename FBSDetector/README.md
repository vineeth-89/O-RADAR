# FBS Detection

This repository contains code of the paper "Gotta Detect ’Em All: Fake Base Station and Multi-Step Attack Detection in Cellular Networks" for detecting Fake Base Stations (FBS) and Multi-Step Attacks (MSAs) from cellular network traces in the User Equipment (UE).

## Repository Structure

```
.
├── codes/
│   ├── classification-models.py
│   ├── cross-validation.py
│   ├── feature-names.py
│   ├── graph_models.py
│   ├── graphsage_metrics.py
│   ├── ml-stats.py
│   ├── stateful-lstm-w-attn.py
│   └── trace-level-classification.py
│
├── dataset/
│   ├── fbs_nas.csv
│   ├── fbs_rrc.csv
│   ├── msa_nas.csv
│   ├── msa_rrc.csv
│   ├── msa_nas_reshaped.csv
│   ├── msa_rrc_reshaped.csv
│   ├── plot_data.json
│
├── phoenix-implementation/
│   ├── dfa.py
│   ├── mm.py
│   ├── packet_utils.py
│   └── pltl.py
│
└── requirements.txt
```

## Description

This project implements a machine learning-based approach for detecting fake base stations and multi-step attacks from cellular network traces.

## Requirements

- Python 3.7+
- PyTorch
- TensorFlow
- NumPy
- Pandas
- scikit-learn
- networkx

See [requirements.txt](requirements.txt) for details.

## Note

See the following sections to reproduce the results in the paper. You can also use [this notebook](run_all.ipynb) to do all of it together.

## Create and Activate Virtual Environment [Optional]

We will create and activate a virtual environment for the project.

Create the virtual environment
```bash
python3 -m venv venv
```

Activate the virtual environment
```bash
source venv/bin/activate
```
## Install Dependencies

```bash
pip install -r requirements.txt
```


## Run Classification Models

The `classification-models.py` script trains and evaluates the following models:
- Random Forest (rf)
- Support Vector Machine (svm)
- Decision Tree (dt)
- XGBoost (xgb)
- K-Nearest Neighbors (knn)
- Naive Bayes (nb)
- Logistic Regression (lr)
- Convolutional Neural Network (cnn)
- Feedforward Neural Network (fnn)
- Long Short-Term Memory Network (lstm)

To run the `classification-models.py` script, use the following command:
```bash
python codes/classification-models.py <[fbs_nas/msa_nas/fbs_rrc/msa_rrc].csv>
```
Replace `<[fbs_nas/msa_nas/fbs_rrc/msa_rrc].csv>` with the path to your dataset file.

Example:
```bash
python codes/classification-models.py dataset/fbs_nas.csv
```

The script will load the dataset, train the models, and display the performance metrics for each model for the provided dataset.

## Run Graph Models

The `graph_models.py` script trains and evaluates the following graph neural network models:
- Graph Attention Network (GAT)
- Graph Attention Network v2 (GATv2)
- Graph Convolutional Network (GCN)
- GraphSAGE
- Graph Transformer

To run the `graph_models.py` script, use the following command:
```bash
python3 codes/graph_models.py <[dataset_path]>
```
Replace `<dataset_path>` with the path to your dataset file.

Example:
```bash
python3 codes/graph_models.py dataset/msa_nas.csv
```

## Stateful LSTM with Attention

To run the `stateful-lstm-w-attn.py` script, use the following command:

```bash
python3 codes/stateful-lstm-w-attn.py <[dataset_path]>
```

Replace `<dataset_path>` with the path to your dataset file.

Example:
```bash
python3 codes/stateful-lstm-w-attn.py dataset/fbs_nas.csv 
```

This script implements our stateful LSTM model with attention mechanism and evaluates its performance.

## Trace-Level Classification
```bash
python3 codes/trace-level-classification.py
```
This script performs trace-level classification using following machine learning models:

- Logistic Regression
- Support Vector Machine
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost

## ML Stats

The `ml-stats.py` script plots various statistics related to the machine learning models.

To run the `ml-stats.py` script, use the following command:
```bash
python3 codes/ml-stats.py
```

This script generates the following plots:
- Accuracy vs Sequence Length for NAS and RRC datasets
- Time Consumption vs Number of Packets
- Memory Consumption vs Number of Packets
- Power Consumption vs Number of Packets

The plots are saved in the `outputs/figures` directory.

## MSA Performance Breakdown (TP, TN, FP, FN)

The `graphsage_metrics.py` script calculates metrics for the GraphSAGE model.

To run the `graphsage_metrics.py` script, use the following command:
```bash
python3 codes/graphsage_metrics.py <[dataset_path]>
```
Replace `<dataset_path>` with the path to your dataset file.

Example:
```bash
python3 codes/graphsage_metrics.py dataset/msa_nas.csv
```

This script generates a tabular summary of the following metrics for each class:
- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)

The results are printed to the console.

## Feature Names
```bash
python3 codes/feature-names.py
```
This script prints the feature names from the dataset in the outputs/column_names_output.txt file.

## Cross-Validation

To run the `cross-validation.py` script, use the following command:
```bash
python3 codes/cross-validation.py <[dataset_path]>
```
Replace `<dataset_path>` with the path to your dataset file.

Example:
```bash
python3 codes/cross-validation.py dataset/msa_nas.csv
```

This script performs leave-one-class-out cross-validation and generates the following outputs:
- Accuracy for each fold
- Detailed results for each fold, including true and predicted labels
- A pivot table summarizing the true and predicted labels across all folds

## Phoenix Implementation

To run our implementation for [PHOENIX's](https://phoenixlte.github.io/) signature-based detection, run the following codes. Download [PHOENIX's](https://phoenixlte.github.io/) signatures and traces from their website and put in the dataset folder.

### DFA

The `dfa.py` script detects anomalies using a Deterministic Finite Automaton (DFA) parsed from a DOT file.

To run the `dfa.py` script, use the following command:
```bash
python phoenix-implementation/dfa.py <state_machine.dot> <trace.pcap>
```
Replace `<state_machine.dot>` with the path to your DOT file and `<trace.pcap>` with the path to your trace file.

Example:
```bash
python3 phoenix-implementation/dfa.py dataset/signatures/dfa/NAS/attach_reject/attach_reject_50_40.trace.dot dataset/NAS_PCAP_logs/attach_reject.pcap
```

### Mealy Machine

The `mm.py` script processes events using a Mealy Machine parsed from a DOT file.

To run the `mm.py` script, use the following command:
```bash
python phoenix-implementation/mm.py <trace.pcap>
```
Replace `<trace.pcap>` with the path to your trace file.

Example:
```bash
python3 phoenix-implementation/mm.py dataset/NAS_PCAP_logs/attach_reject.pcap
```

### PLTL

The `pltl.py` script checks events against Propositional Linear Temporal Logic (PLTL) signatures.

To run the `pltl.py` script, use the following command:
```bash
python phoenix-implementation/pltl.py <trace.pcap>
```
Replace `<trace.pcap>` with the path to your trace file.

Example:
```bash
python phoenix-implementation/pltl.py dataset/NAS_PCAP_logs/attach_reject.pcap
```

## Citation

If you use this dataset, models, or code modules, please cite the following paper:

```
@misc{mubasshir2025gottadetectemall,
      title={Gotta Detect 'Em All: Fake Base Station and Multi-Step Attack Detection in Cellular Networks}, 
      author={Kazi Samin Mubasshir and Imtiaz Karim and Elisa Bertino},
      year={2025},
      eprint={2401.04958},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2401.04958}, 
}
```
