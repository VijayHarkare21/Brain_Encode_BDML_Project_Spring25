#!/usr/bin/env python3
"""
Script for training and evaluating classifiers on EEG embeddings for both image and text data.

This script supports:
- Two types of EEG embeddings: CBraMod and LaBraM
- Two ways of handling multi-subject data: averaging or stacking with subject encoding
- Multiple classifier types: Decision Tree, Logistic Regression, Softmax, MLP, Naive Bayes, KNN
- Support for both image and text classification tasks

Usage:
python eeg_classifier.py \
    --mode image \
    --eeg_dir path/to/eeg_images \
    --embedding_type cbramod \
    --subject_handling average \
    --classifier_type logistic \
    --output_dir results \
    --image_metadata path/to/image_metadata.npy \
    --text_csv path/to/text_categories.csv \
    [--batch_size 32] \
    [--random_seed 42] \
    [--cv_folds 5]
"""

import os
import argparse
import pickle
import hashlib
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_eeg_image_data(directory: str) -> Dict[str, dict]:
    """Load EEG image data from .npy files in a directory."""
    print(f"[INFO] Loading EEG image data from: {directory}")
    data = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.npy'):
            continue
        if 'test' in fname.lower():
            continue
        
        key = os.path.splitext(fname)[0]
        print(f"[INFO]   Loading train file: {fname}")
        data[key] = np.load(os.path.join(directory, fname), allow_pickle=True).item()
    return data


def load_eeg_text_data(directory: str) -> Dict[str, List[dict]]:
    """Load EEG text data from .npy files in a directory."""
    print(f"[INFO] Loading EEG text data from: {directory}")
    data = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.npy'):
            continue
        
        print(f"[INFO]   Loading file: {fname}")
        arr = np.load(os.path.join(directory, fname), allow_pickle=True).item()
        subject = next(iter(arr.keys()))
        data[subject] = arr[subject]
    return data


def load_image_labels(metadata_path: str, things_map_path: str) -> Dict[str, str]:
    """Load image labels from metadata file."""
    print(f"[INFO] Loading image labels from {metadata_path}")
    print(f"[INFO] Loading high-level image labels from {things_map_path}")
    meta = np.load(metadata_path, allow_pickle=True).item()
    things_map = pd.read_csv(things_map_path, delimiter="\t")
    files = meta['train_img_files']
    concepts = meta['train_img_concepts']
    things_concepts = meta['train_img_concepts_THINGS']
    
    # Create a mapping from full path to concept label
    path_to_label = {}
    for things_concept, concept, fname in zip(things_concepts, concepts, files):
        # print(things_concept.split("_")[0])
        # print(things_map.iloc[int(things_concept.split("_")[0]) + 1])
        row = things_map.iloc[int(things_concept.split("_")[0]) - 1]
        high_concept = str(things_map.columns[row == 1][0]) if not (row == 0).all() else 'miscellaneous'
        path_key = os.path.join(concept, fname)
        path_to_label[path_key] = high_concept
        
    return path_to_label


# def load_text_labels(csv_path: str) -> Dict[str, str]:
#     """Load text labels from CSV file."""
#     print(f"[INFO] Loading text labels from {csv_path}")
#     df = pd.read_csv(csv_path, delimiter=";")
#     # Map sentences to their categories
#     return dict(zip(df.iloc[:, 2], df['category']))
def load_text_labels(csv_path: str) -> Dict[str, str]:
    print(f"[INFO] Loading text labels from {csv_path}")
    df = pd.read_csv(csv_path, delimiter=";")
    # assume the sentence is in column 2
    sentences = df.iloc[:, 2].astype(str).str.strip().str.lower()
    categories = df["category"].astype(str).str.strip()
    return dict(zip(sentences, categories))


def extract_image_path_key(full_path: str) -> str:
    """Extract concept/filename from a full path."""
    parts = full_path.split(os.sep)
    # Get the last two parts: concept and filename
    if len(parts) >= 2:
        return os.path.join(parts[-2], parts[-1])
    return full_path


def aggregate_eeg_image(
    img_data: Dict[str, dict],
    eeg_type: str,
    subject_handling: str = "average",
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Aggregate EEG image embeddings across subjects.
    
    Args:
        img_data: Dictionary of EEG data per subject
        eeg_type: Type of EEG embedding (cbramod or labram)
        subject_handling: How to handle multiple subjects (average or stack)
        
    Returns:
        embeddings: Aggregated embeddings
        paths: Image paths
        subject_ids: List of subject IDs (if subject_handling is "stack")
    """
    key = f"embeds_{eeg_type}"
    by_path = defaultdict(list)
    subject_ids = sorted(img_data.keys())
    subject_path_pairs = []
    
    # Collect embeddings by path
    for subj, arr in img_data.items():
        if "img_paths" not in arr or key not in arr:
            print(f"[WARNING] Subject {subj} missing '{key}', skipping.")
            continue
            
        paths = arr["img_paths"]
        embs = arr[key]
        
        if len(paths) != len(embs):
            raise ValueError(f"Length mismatch for subject {subj}: "
                            f"{len(paths)} paths vs {len(embs)} embeds.")
                            
        for p, e in zip(paths, embs):
            by_path[p].append(e)
            if subject_handling == "stack":
                subject_path_pairs.append((subj, p))
                
    if not by_path:
        raise ValueError(f"No EEG-image embeddings found for type '{eeg_type}'")
    
    # Aggregate embeddings
    paths = sorted(by_path.keys())
    
    if subject_handling == "average":
        embeddings = np.vstack([np.mean(by_path[p], axis=0) for p in paths])
        return embeddings, paths, []
    else:  # stack
        # For stack, we maintain one embedding per subject-path pair
        embeddings = []
        final_paths = []
        final_subjects = []
        
        for subj, path in subject_path_pairs:
            embedding_idx = subject_ids.index(subj)
            path_idx = paths.index(path)
            embeddings.append(by_path[path][embedding_idx])
            final_paths.append(path)
            final_subjects.append(subj)
            
        return np.vstack(embeddings), final_paths, final_subjects

def aggregate_eeg_text(
    txt_data: Dict[str, List[dict]],
    eeg_type: str,
    subject_handling: str = "average",
) -> Tuple[np.ndarray, List[str], List[str]]:
    # 1) decide on the set of sentences
    subj_ids      = sorted(txt_data.keys())
    per_subject   = [
        {t["content"] for t in txt_data[sid] if t and f"embeds_{eeg_type}" in t}
        for sid in subj_ids
    ]
    if subject_handling == "average":
        common = sorted(set.intersection(*per_subject))
    else:
        common = sorted(set.union(*per_subject))
    if not common:
        raise ValueError("No sentences found in common.")

    # 2) figure out feature dimension from first real embedding
    for sid in subj_ids:
        for t in txt_data[sid]:
            if t and f"embeds_{eeg_type}" in t:
                feature_dim = np.asarray(t[f"embeds_{eeg_type}"]).ravel().shape[0]
                break
        else:
            continue
        break
    else:
        raise ValueError(f"No embeddings of type {eeg_type} found anywhere!")

    # 3) per‑subject, per‑sentence mean (always produce a flat vector)
    subj_sent_emb = {sid: {} for sid in subj_ids}
    for sid in subj_ids:
        for sent in common:
            sent_embs = [
                np.asarray(t[f"embeds_{eeg_type}"]).ravel()
                for t in txt_data[sid]
                if t and t["content"] == sent and f"embeds_{eeg_type}" in t
            ]
            if not sent_embs:
                sent_embs = [np.zeros(feature_dim)]
            subj_sent_emb[sid][sent] = np.mean(sent_embs, axis=0)

    # 4) build final matrix
    if subject_handling == "average":
        # one row per sentence, averaged across subjects
        embeddings = np.vstack([
            np.mean(np.stack([subj_sent_emb[sid][sent] for sid in subj_ids], axis=0),
                    axis=0)
            for sent in common
        ])
        return embeddings, common, []
    else:
        # one row per (subject, sentence)
        rows, contents, sids = [], [], []
        for sid in subj_ids:
            for sent in common:
                rows.append(subj_sent_emb[sid][sent])
                contents.append(sent)
                sids.append(sid)
        return np.vstack(rows), contents, sids

# def aggregate_eeg_text(
#     txt_data: Dict[str, List[dict]],
#     eeg_type: str,
#     subject_handling: str = "average",
# ) -> Tuple[np.ndarray, List[str], List[str]]:
#     """
#     Aggregate EEG text embeddings across subjects.
    
#     Args:
#         txt_data: Dictionary of EEG data per subject
#         eeg_type: Type of EEG embedding (cbramod or labram)
#         subject_handling: How to handle multiple subjects (average or stack)
        
#     Returns:
#         embeddings: Aggregated embeddings
#         contents: Text contents
#         subject_ids: List of subject IDs (if subject_handling is "stack")
#     """
#     # 1) identify sentences seen by all subjects
#     subj_ids = sorted(txt_data.keys())
#     per_subject_sets = []
    
#     for sid in subj_ids:
#         seen = {t["content"] for t in txt_data[sid] if t and f"embeds_{eeg_type}" in t}
#         per_subject_sets.append(seen)
        
#     # common = sorted(set.intersection(*per_subject_sets))
#     common = sorted(set.union(*per_subject_sets))
#     print(type(common))
#     print(len(common))
#     print(common[0])
#     if not common:
#         raise ValueError("No sentences found.")

#     # 2) average repeated trials per subject
#     subj_sent_emb = {sid: {} for sid in subj_ids}
#     for sid in subj_ids:
#         by_sent = defaultdict(list)
#         for t in txt_data[sid]:
#             if t and f"embeds_{eeg_type}" in t and t["content"] in common:
#                 by_sent[t["content"]].append(t[f"embeds_{eeg_type}"])
#             elif t and t["content"] in common:
#                 if eeg_type == 'cbramod':
#                     by_sent[t["content"]].append(np.zeros((600,)))
#                 else:
#                     by_sent[t["content"]].append(np.zeros((200,)))
#         for sent in common:
#             subj_sent_emb[sid][sent] = np.mean(by_sent[sent], axis=0)

#     # 3) build final matrix
#     if subject_handling == "average":
#         embeddings = np.vstack([
#             np.mean([subj_sent_emb[sid][sent] for sid in subj_ids], axis=0)
#             for sent in common
#         ])
#         return embeddings, common, []
#     else:  # stack
#         # One row per subject×sentence
#         embeddings = []
#         contents = []
#         subject_ids = []
        
#         for sid in subj_ids:
#             for sent in common:
#                 embeddings.append(subj_sent_emb[sid][sent])
#                 contents.append(sent)
#                 subject_ids.append(sid)
                
#         return np.vstack(embeddings), contents, subject_ids


def encode_subjects(subject_ids: List[str]) -> np.ndarray:
    """One-hot encode subject IDs."""
    encoder = OneHotEncoder(sparse_output=False)
    return encoder.fit_transform(np.array(subject_ids).reshape(-1, 1))


def prepare_data_for_classification(
    embeddings: np.ndarray,
    items: List[str],
    labels_dict: Dict[str, str],
    subject_ids: Optional[List[str]] = None,
    extract_path_key_fn: Optional[callable] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    """
    Prepare EEG embedding data for classification.
    
    Args:
        embeddings: EEG embeddings
        items: List of paths or text content
        labels_dict: Dictionary mapping items to labels
        subject_ids: List of subject IDs (for stacked embeddings)
        extract_path_key_fn: Function to extract key from path (for images)
        test_size: Fraction of data to use for testing
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test: Train/test split
        label_encoder: Fitted LabelEncoder for labels
    """
    # Get labels for each item
    # y_raw = []
    # valid_indices = []
    
    # for i, item in enumerate(items):
    #     key = extract_path_key_fn(item) if extract_path_key_fn else item
    #     if key in labels_dict:
    #         y_raw.append(labels_dict[key])
    #         valid_indices.append(i)
    #     else:
    #         print(f"[WARNING] No label found for item: {key}")
    y_raw = []
    valid_indices = []
    for i, item in enumerate(items):
        key = extract_path_key_fn(item) if extract_path_key_fn else item
        key_norm = key.strip().lower()
        if key_norm in labels_dict:
            y_raw.append(labels_dict[key_norm])
            valid_indices.append(i)
        else:
            print(f"[WARNING] No label found for item: {key!r}")
    
    # Filter embeddings and subject_ids to only include items with labels
    X = embeddings[valid_indices]
    if subject_ids:
        subject_ids_filtered = [subject_ids[i] for i in valid_indices]
        subject_onehot = encode_subjects(subject_ids_filtered)
        X = np.hstack([X, subject_onehot])
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, label_encoder


def train_evaluate_sklearn_classifier(
    clf,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    classifier_name: str,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """Train and evaluate a scikit-learn classifier."""
    print(f"[INFO] Training {classifier_name}...")
    print(X_train.shape)
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation on training set
    cv_scores = cross_val_score(
        clf, X_train, y_train, cv=StratifiedKFold(n_splits=cv_folds), scoring='accuracy'
    )
    
    # Classification report
    # report = classification_report(
    #     y_test, 
    #     y_pred, 
    #     target_names=label_encoder.classes_,
    #     output_dict=True
    # )
    # figure out which encoded labels actually appear
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    # turn those back into string names
    present_names = label_encoder.inverse_transform(unique_labels)
    
    report = classification_report(
        y_test,
        y_pred,
        labels=unique_labels,        # numeric labels that appear
        target_names=present_names,  # their human‑readable names
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'classifier': classifier_name,
        'accuracy': accuracy,
        'cv_scores': cv_scores,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'classification_report': report,
        'confusion_matrix': cm,
        'model': clf
    }
    
    print(f"[INFO] {classifier_name} - Test accuracy: {accuracy:.4f}, "
          f"CV accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    return results


# PyTorch models
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.2):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


def train_pytorch_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 0.001,
    device: str = 'cpu',
    early_stopping: int = 10,
) -> Dict[str, Any]:
    """Train a PyTorch model on EEG embeddings."""
    print(X_train.shape)
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and optimizer
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_accuracy = 0.0
    best_epoch = 0
    no_improve_count = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = correct / total
        
        train_losses.append(epoch_loss)
        val_accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= early_stopping:
            print(f'[INFO] Early stopping at epoch {epoch+1}')
            break
    
    print(f'[INFO] Best accuracy: {best_accuracy:.4f} at epoch {best_epoch+1}')
    
    # Final evaluation
    model.eval()
    y_pred = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
    
    return {
        'accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'y_pred': np.array(y_pred),
        'model': model
    }


def save_results(results: Dict[str, Any], output_dir: str, prefix: str):
    """Save classification results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results summary
    summary = {
        'classifier': results['classifier'],
        'accuracy': results['accuracy'],
        'cv_mean': results.get('cv_mean'),
        'cv_std': results.get('cv_std'),
    }
    
    summary_file = os.path.join(output_dir, f"{prefix}_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_file, index=False)
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, f"{prefix}_detailed.pkl")
    with open(detailed_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Plot confusion matrix if available
    if 'confusion_matrix' in results:
        cm = results['confusion_matrix']
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {results['classifier']}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"))
        plt.close()
    
    # Plot training curves for PyTorch models
    if 'train_losses' in results:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(results['train_losses'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(results['val_accuracies'])
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_training_curves.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="EEG Embedding Classifier")
    
    # Required arguments
    parser.add_argument("--mode", type=str, required=True, choices=["image", "text", "both"],
                        help="Classification mode: image, text, or both")
    parser.add_argument("--eeg_dir", type=str, required=True,
                        help="Directory containing EEG embeddings")
    parser.add_argument("--embedding_type", type=str, required=True, 
                        choices=["cbramod", "labram", "both"],
                        help="Type of EEG embedding to use")
    parser.add_argument("--subject_handling", type=str, required=True,
                        choices=["average", "stack"],
                        help="How to handle multiple subjects: average or stack with encoding")
    parser.add_argument("--classifier_type", type=str, required=True,
                        choices=["decision_tree", "logistic", "softmax", "mlp", 
                               "naive_bayes", "knn", "all"],
                        help="Type of classifier to train")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results")
    
    # Optional arguments
    parser.add_argument("--image_metadata", type=str,
                        help="Path to image metadata .npy file")
    parser.add_argument("--image_eeg_dir", type=str,
                        help="Directory containing image EEG embeddings (if different from eeg_dir)")
    parser.add_argument("--things_map_path", type=str,
                        help="Path to THINGS label map")
    parser.add_argument("--text_csv", type=str,
                        help="Path to text categories CSV file")
    parser.add_argument("--text_eeg_dir", type=str,
                        help="Directory containing text EEG embeddings (if different from eeg_dir)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for PyTorch models")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to use for testing")
    parser.add_argument("--mlp_hidden_dims", type=str, default="256,128",
                        help="Comma-separated hidden layer dimensions for MLP")
    parser.add_argument("--knn_neighbors", type=int, default=5,
                        help="Number of neighbors for KNN")
    parser.add_argument("--scale_features", action="store_true",
                        help="Scale features using StandardScaler")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process embedding types
    embedding_types = []
    if args.embedding_type == "both":
        embedding_types = ["cbramod", "labram"]
    else:
        embedding_types = [args.embedding_type]
    
    # Process classifier types
    classifier_types = []
    if args.classifier_type == "all":
        classifier_types = ["decision_tree", "logistic", "softmax", "mlp", "naive_bayes", "knn"]
    else:
        classifier_types = [args.classifier_type]
    
    # Parse MLP hidden dimensions
    mlp_hidden_dims = [int(dim) for dim in args.mlp_hidden_dims.split(",")]
    
    # Determine device for PyTorch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    
    # IMAGE CLASSIFICATION
    if args.mode in ("image", "both"):
        if not args.image_metadata:
            parser.error("--image_metadata is required for image classification")
            
        image_eeg_dir = args.image_eeg_dir if args.image_eeg_dir else args.eeg_dir
        img_data = load_eeg_image_data(image_eeg_dir)
        img_labels = load_image_labels(args.image_metadata, args.things_map_path)
        
        for et in embedding_types:
            print(f"\n[INFO] Processing image classification with {et} embeddings")
            
            # Get embeddings
            embeddings, paths, subject_ids = aggregate_eeg_image(
                img_data, et, args.subject_handling
            )
            
            # Prepare data
            X_train, X_test, y_train, y_test, label_encoder = prepare_data_for_classification(
                embeddings, 
                paths, 
                img_labels, 
                subject_ids if args.subject_handling == "stack" else None,
                extract_path_key_fn=extract_image_path_key,
                test_size=args.test_size,
                random_state=args.random_seed
            )
            
            if args.subject_handling == "stack":
                # fraction of each split to keep
                f = 0.1    # e.g. 0.3 → keep 30% of each of train and test
                rng = np.random.RandomState(args.random_seed)
                
                def subsample(X, y, frac, rng):
                    n = len(y)
                    keep = int(n * frac)
                    idx = rng.choice(n, size=keep, replace=False)
                    return X[idx], y[idx]
                
                # subsample train and test separately
                if f < 1.0:
                    X_train, y_train = subsample(X_train, y_train, f, rng)
                    X_test,  y_test  = subsample(X_test,  y_test,  f, rng)
            
            # Scale features if requested
            if args.scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            num_classes = len(label_encoder.classes_)
            print(f"[INFO] Number of image classes: {num_classes}")
            
            # Train and evaluate classifiers
            for clf_type in classifier_types:
                result_prefix = f"image_{et}_{args.subject_handling}_{clf_type}"
                
                if clf_type == "decision_tree":
                    clf = DecisionTreeClassifier(random_state=args.random_seed)
                    results = train_evaluate_sklearn_classifier(
                        clf, X_train, X_test, y_train, y_test, label_encoder, 
                        "Decision Tree", args.cv_folds
                    )
                    
                elif clf_type == "logistic":
                    clf = LogisticRegression(
                        multi_class='multinomial', 
                        max_iter=1000, 
                        random_state=args.random_seed
                    )
                    results = train_evaluate_sklearn_classifier(
                        clf, X_train, X_test, y_train, y_test, label_encoder, 
                        "Logistic Regression", args.cv_folds
                    )
                    
                elif clf_type == "naive_bayes":
                    clf = GaussianNB()
                    results = train_evaluate_sklearn_classifier(
                        clf, X_train, X_test, y_train, y_test, label_encoder, 
                        "Gaussian Naive Bayes", args.cv_folds
                    )
                    
                elif clf_type == "knn":
                    clf = KNeighborsClassifier(n_neighbors=args.knn_neighbors)
                    results = train_evaluate_sklearn_classifier(
                        clf, X_train, X_test, y_train, y_test, label_encoder, 
                        f"KNN (k={args.knn_neighbors})", args.cv_folds
                    )
                    
                elif clf_type == "softmax":
                    model = SoftmaxClassifier(X_train.shape[1], num_classes)
                    pt_results = train_pytorch_model(
                        model, X_train, X_test, y_train, y_test, 
                        batch_size=args.batch_size, device=device
                    )
                    results = {
                        'classifier': 'Softmax',
                        'accuracy': pt_results['accuracy'],
                        'y_pred': pt_results['y_pred'],
                        'train_losses': pt_results['train_losses'],
                        'val_accuracies': pt_results['val_accuracies'],
                        'model': pt_results['model'],
                        'classification_report': classification_report(
                            y_test, pt_results['y_pred'], 
                            target_names=label_encoder.classes_,
                            output_dict=True
                        ),
                        'confusion_matrix': confusion_matrix(y_test, pt_results['y_pred'])
                    }
                    
                elif clf_type == "mlp":
                    model = MLPClassifier(X_train.shape[1], mlp_hidden_dims, num_classes)
                    pt_results = train_pytorch_model(
                        model, X_train, X_test, y_train, y_test, 
                        batch_size=args.batch_size, device=device
                    )
                    results = {
                        'classifier': 'MLP',
                        'accuracy': pt_results['accuracy'],
                        'y_pred': pt_results['y_pred'],
                        'train_losses': pt_results['train_losses'],
                        'val_accuracies': pt_results['val_accuracies'],
                        'model': pt_results['model'],
                        'classification_report': classification_report(
                            y_test, pt_results['y_pred'], 
                            target_names=label_encoder.classes_,
                            output_dict=True
                        ),
                        'confusion_matrix': confusion_matrix(y_test, pt_results['y_pred'])
                    }
                
                # Save results
                save_results(results, args.output_dir, result_prefix)
    
    # TEXT CLASSIFICATION
    if args.mode in ("text", "both"):
        if not args.text_csv:
            parser.error("--text_csv is required for text classification")
            
        text_eeg_dir = args.text_eeg_dir if args.text_eeg_dir else args.eeg_dir
        txt_data = load_eeg_text_data(text_eeg_dir)
        txt_labels = load_text_labels(args.text_csv)
        
        for et in embedding_types:
            print(f"\n[INFO] Processing text classification with {et} embeddings")
            
            # Get embeddings
            embeddings, contents, subject_ids = aggregate_eeg_text(
                txt_data, et, args.subject_handling
            )
            
            # Prepare data
            X_train, X_test, y_train, y_test, label_encoder = prepare_data_for_classification(
                embeddings, 
                contents, 
                txt_labels, 
                subject_ids if args.subject_handling == "stack" else None,
                test_size=args.test_size,
                random_state=args.random_seed
            )
            
            # Scale features if requested
            if args.scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            num_classes = len(label_encoder.classes_)
            print(f"[INFO] Number of text classes: {num_classes}")
            
            # Train and evaluate classifiers
            for clf_type in classifier_types:
                result_prefix = f"text_{et}_{args.subject_handling}_{clf_type}"
                
                if clf_type == "decision_tree":
                    clf = DecisionTreeClassifier(random_state=args.random_seed)
                    results = train_evaluate_sklearn_classifier(
                        clf, X_train, X_test, y_train, y_test, label_encoder, 
                        "Decision Tree", args.cv_folds
                    )
                    
                elif clf_type == "logistic":
                    clf = LogisticRegression(
                        multi_class='multinomial', 
                        max_iter=1000, 
                        random_state=args.random_seed
                    )
                    results = train_evaluate_sklearn_classifier(
                        clf, X_train, X_test, y_train, y_test, label_encoder, 
                        "Logistic Regression", args.cv_folds
                    )
                    
                elif clf_type == "naive_bayes":
                    clf = GaussianNB()
                    results = train_evaluate_sklearn_classifier(
                        clf, X_train, X_test, y_train, y_test, label_encoder, 
                        "Gaussian Naive Bayes", args.cv_folds
                    )
                    
                elif clf_type == "knn":
                    clf = KNeighborsClassifier(n_neighbors=args.knn_neighbors)
                    results = train_evaluate_sklearn_classifier(
                        clf, X_train, X_test, y_train, y_test, label_encoder, 
                        f"KNN (k={args.knn_neighbors})", args.cv_folds
                    )
                    
                elif clf_type == "softmax":
                    model = SoftmaxClassifier(X_train.shape[1], num_classes)
                    pt_results = train_pytorch_model(
                        model, X_train, X_test, y_train, y_test, 
                        batch_size=args.batch_size, device=device
                    )
                    unique_labels = np.unique(np.concatenate([y_test, pt_results['y_pred']]))
                    # turn those back into string names
                    present_names = label_encoder.inverse_transform(unique_labels)
                    results = {
                        'classifier': 'Softmax',
                        'accuracy': pt_results['accuracy'],
                        'y_pred': pt_results['y_pred'],
                        'train_losses': pt_results['train_losses'],
                        'val_accuracies': pt_results['val_accuracies'],
                        'model': pt_results['model'],
                        'classification_report': classification_report(
                            y_test, pt_results['y_pred'], 
                            labels=unique_labels,
                            target_names=present_names,
                            output_dict=True
                        ),
                        'confusion_matrix': confusion_matrix(y_test, pt_results['y_pred'])
                    }
                    
                elif clf_type == "mlp":
                    model = MLPClassifier(X_train.shape[1], mlp_hidden_dims, num_classes)
                    pt_results = train_pytorch_model(
                        model, X_train, X_test, y_train, y_test, 
                        batch_size=args.batch_size, device=device
                    )
                    unique_labels = np.unique(np.concatenate([y_test, pt_results['y_pred']]))
                    # turn those back into string names
                    present_names = label_encoder.inverse_transform(unique_labels)
                    results = {
                        'classifier': 'MLP',
                        'accuracy': pt_results['accuracy'],
                        'y_pred': pt_results['y_pred'],
                        'train_losses': pt_results['train_losses'],
                        'val_accuracies': pt_results['val_accuracies'],
                        'model': pt_results['model'],
                        'classification_report': classification_report(
                            y_test, pt_results['y_pred'], 
                            labels=unique_labels,
                            target_names=present_names,
                            output_dict=True
                        ),
                        'confusion_matrix': confusion_matrix(y_test, pt_results['y_pred'])
                    }
                
                # Save results
                save_results(results, args.output_dir, result_prefix)
    
    print(f"\n[INFO] All classifications completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()