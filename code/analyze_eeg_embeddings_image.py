import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

parser = argparse.ArgumentParser(description="Analyze EEG embeddings")
parser.add_argument('--embed_type', type=str, default='cbramod', help='Type of embedding to analyze (labram, cbramod)')
parser.add_argument('--subject', type=int, default=1, help='Subject ID to analyze')

args = parser.parse_args()
embed_type = args.embed_type
subject = args.subject

complete_data = np.load(r"D:\Vijay\NYU\Spring_25\BDMLS\Project\dataset\ImageNetEEG\processed_eeg_signals.npy", allow_pickle=True)
complete_data = complete_data.item()
subject_indices = [x for x, elem in enumerate(complete_data['rawData']) if elem['subject'] == subject]

del complete_data

embedding_data = np.load(rf"D:\Vijay\NYU\Spring_25\BDMLS\Project\dataset\ImageNetEEG\processed_eeg_embeds_{embed_type}.npy", allow_pickle=True).item()[f'embeds_{embed_type}']

# only keep the embeddings for the subject
embedding_data = [embedding_data[i] for i in subject_indices]

save_dir = rf"D:\Vijay\NYU\Spring_25\BDMLS\Project\code\figs\imageEEG\{subject}\embedding_analysis\{embed_type}"
print(f"Save directory: {save_dir}")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

embeddings_np = np.array(embedding_data)
print("Embeddings shape before squeeze:", embeddings_np.shape)

embeddings_np = embeddings_np.squeeze(axis=1)

print("Embeddings shape:", embeddings_np.shape)

def variance_analysis(embeddings, embed_type):
    reshaped_embeddings = np.array(embeddings)

    variances = np.var(reshaped_embeddings, axis=0)
    low_variance_threshold = 1e-6  # Adjust as needed
    low_variance_features = np.where(variances < low_variance_threshold)[0]

    print(f"Number of features with low variance: {len(low_variance_features)}")
    # print(f"Indices of low variance features: {low_variance_features}")

def correlation_analysis(embeddings, embed_type):
    reshaped_embeddings = np.array(embeddings)

    correlation_matrix = np.corrcoef(reshaped_embeddings.T)

    # plt.figure()
    plt.imshow(correlation_matrix, cmap='viridis')
    plt.colorbar()
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(save_dir, f"correlation_matrix_{embed_type}.png"))
    plt.show()

    high_correlation_threshold = 0.9  # Adjust as needed
    high_correlation_pairs = np.where(np.abs(correlation_matrix) > high_correlation_threshold)
    high_correlation_pairs = [(i, j) for i, j in zip(high_correlation_pairs[0], high_correlation_pairs[1]) if i != j]

    print(f"Number of highly correlated feature pairs: {len(high_correlation_pairs)}")
    # print(f"Highly correlated feature pairs: {high_correlation_pairs}")

def pca_analysis(embeddings, embed_type):
    # Eigenvector analysis (examine the top eigenvectors for feature importance)
    # This is more relevant for understanding the directions of variance
    # in the original feature space.

    pca = PCA()
    reshaped_embeddings = np.array(embeddings)
    print(reshaped_embeddings.shape)
    pca.fit(reshaped_embeddings)

    explained_variance_ratio = pca.explained_variance_ratio_

    print(f"Explained variance ratio of first few components: {explained_variance_ratio[:10]}")

    # cumulative_variance = np.cumsum(explained_variance_ratio)
    # plt.figure()
    plt.plot(explained_variance_ratio)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Explained Variance Ratio vs. Number of Components")
    plt.savefig(os.path.join(save_dir, f"explained_variance_ratio_{embed_type}.png"))
    plt.show()

    # Eigenvalue and Eigenvector Analysis
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_

    print(f"Eigenvalues shape: {eigenvalues.shape}")
    print(f"Eigenvectors shape: {eigenvectors.shape}")

    # Analyze eigenvalues (e.g., plot or check the distribution)
    # plt.figure()
    plt.plot(eigenvalues)
    plt.xlabel("Principal Component Index")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalue Distribution")
    plt.savefig(os.path.join(save_dir, f"eigenvalue_distribution_{embed_type}.png"))
    plt.show()
    num_top_eigenvectors = 5  # Analyze the top 5 eigenvectors (adjust as needed)
    for i in range(min(num_top_eigenvectors, eigenvectors.shape[0])):
        top_eigenvector = eigenvectors[i]
        plt.plot(top_eigenvector)
        plt.xlabel("Feature Index")
        plt.ylabel("Weight")
        plt.title(f"Eigenvector {i+1} - Feature Weights")
        plt.savefig(os.path.join(save_dir, f"eigenvector_{i+1}_feature_weights_{embed_type}.png"))
        plt.show()
            

if args.embed_type == "cbramod":
    embeddings = embeddings_np
    variance_analysis(embeddings, "cbramod")
    correlation_analysis(embeddings, "cbramod")
    pca_analysis(embeddings, "cbramod")
elif args.embed_type == "labram":
    embeddings = embeddings_np
    variance_analysis(embeddings, "labram")
    correlation_analysis(embeddings, "labram")
    pca_analysis(embeddings, "labram")