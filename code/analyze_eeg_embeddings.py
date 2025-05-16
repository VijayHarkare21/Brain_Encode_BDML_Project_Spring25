import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

parser = argparse.ArgumentParser(description="Analyze EEG embeddings")
parser.add_argument('--embed_type', type=str, default='cbramod', help='Type of embedding to analyze (labram, cbramod)')
parser.add_argument('--subject', type=str, default='YAC', help='Subject ID to analyze')

args = parser.parse_args()
embed_type = args.embed_type
subject = args.subject

eeg_path = rf"D:\Vijay\NYU\Spring_25\BDMLS\Project\dataset\ZuCo\task2-NR-2.0\pickle\task2-NR-2.0-dataset_{subject}_embeds_clustered_new.npy"

data = np.load(eeg_path, allow_pickle=True).item()
print(f"Loaded data from {eeg_path}")

sentence_data = data[subject]
print(f"Loaded sentence data for subject {subject}")

save_dir = rf"D:\Vijay\NYU\Spring_25\BDMLS\Project\code\figs\{subject}\embedding_analysis\{embed_type}"
print(f"Save directory: {save_dir}")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

embeddings = []
valid_indices = []

# Filter out None embeddings and compute mean across dimension 1
for i, data_dict in enumerate(sentence_data):
    if data_dict is not None:
        embed = data_dict.get(f"embeds_{embed_type}")
        if embed is not None:
            embeddings.append(embed)
            # print(i, embed.shape)
            valid_indices.append(i)

if not embeddings:
    print("No valid embeddings found.")
    exit(1)

embeddings_np = np.array(embeddings)

embeddings_np = embeddings_np.squeeze(axis=1)

print("Embeddings shape:", embeddings_np.shape)

def variance_analysis(embeddings, embed_type):
    if embed_type == "cbramod":
        # Reshape to (N, 105, 200) -> (N, 105 * 200)
        # reshaped_embeddings = np.array([emb.reshape(105 * 200) for emb in embeddings])
        reshaped_embeddings = np.mean(embeddings, axis=1)
    else:  # labram
        reshaped_embeddings = np.array(embeddings)

    variances = np.var(reshaped_embeddings, axis=0)
    low_variance_threshold = 1e-6  # Adjust as needed
    low_variance_features = np.where(variances < low_variance_threshold)[0]

    print(f"Number of features with low variance: {len(low_variance_features)}")
    # print(f"Indices of low variance features: {low_variance_features}")

    if embed_type == "cbramod":
        # Analyze variance per channel
        channel_variances = np.var(embeddings, axis=0).mean(axis=1) # Average variance across features for each channel
        # plt.figure()
        plt.plot(channel_variances)
        plt.xlabel("EEG Channel")
        plt.ylabel("Average Variance")
        plt.title("Average Variance per EEG Channel (cbramod)")
        plt.savefig(os.path.join(save_dir, f"variance_per_channel_{embed_type}.png"))
        plt.show()

def correlation_analysis(embeddings, embed_type):
    if embed_type == "cbramod":
        # reshaped_embeddings = np.array([emb.reshape(105 * 200) for emb in embeddings])
        reshaped_embeddings = np.mean(embeddings, axis=1)
    else:  # labram
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

    if embed_type == "cbramod":
        # Analyze correlation per channel
        channel_correlation_matrices = []
        for i in range(105):
            channel_data = embeddings[:, i, :]
            channel_correlation_matrices.append(np.corrcoef(channel_data.T))

        # Summarize channel correlations
        average_channel_correlations = []
        for corr_matrix in channel_correlation_matrices:
            # Calculate the average of the upper triangle (excluding the diagonal)
            upper_triangle = np.triu(corr_matrix, k=1)
            average_correlation = np.mean(upper_triangle)
            average_channel_correlations.append(average_correlation)

        # Visualize average channel correlations
        # plt.figure()
        plt.plot(average_channel_correlations)
        plt.xlabel("EEG Channel")
        plt.ylabel("Average Correlation")
        plt.title("Average Correlation per EEG Channel (cbramod)")
        plt.savefig(os.path.join(save_dir, f"average_correlation_per_channel_{embed_type}.png"))
        plt.show()
    

        # You can also print some statistics
        print(f"Average correlation across all channels: {np.mean(average_channel_correlations)}")
        print(f"Max correlation: {np.max(average_channel_correlations)}")
        print(f"Min correlation: {np.min(average_channel_correlations)}")

def pca_analysis(embeddings, embed_type):
    # Eigenvector analysis (examine the top eigenvectors for feature importance)
    # This is more relevant for understanding the directions of variance
    # in the original feature space.

    if embed_type == 'labram':
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
    else:
        pca = PCA()
        reshaped_embeddings = np.array(embeddings)
        reshaped_embeddings = reshaped_embeddings.reshape(reshaped_embeddings.shape[0], -1)  # Flatten the last two dimensions
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

        num_top_eigenvectors = 5
        for i in range(min(num_top_eigenvectors, eigenvectors.shape[0])):
            top_eigenvector = eigenvectors[i]
            # Reshape eigenvector to (105, 200) for channel-feature interpretation
            top_eigenvector_reshaped = top_eigenvector.reshape(105, 200)
            
            # Example: Calculate average absolute weight per channel
            channel_weights = np.mean(np.abs(top_eigenvector_reshaped), axis=1)
            
            # plt.figure()
            plt.plot(channel_weights)
            plt.xlabel("EEG Channel")
            plt.ylabel("Average Absolute Weight")
            plt.title(f"Eigenvector {i+1} - Average Channel Weights")
            plt.savefig(os.path.join(save_dir, f"eigenvector_{i+1}_average_channel_weights_{embed_type}.png"))
            plt.show()
            
            # # Example: Visualize weights across features for a specific channel
            # # (e.g., channel 0)
            # # plt.figure()
            # plt.plot(top_eigenvector_reshaped[0])
            # plt.xlabel("Feature Index")
            # plt.ylabel("Weight")
            # plt.title(f"Eigenvector {i+1} - Feature Weights (Channel 0)")
            # plt.savefig(os.path.join(save_dir, f"eigenvector_{i+1}_feature_weights_channel_0_{embed_type}.png"))
            # plt.show()
            

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