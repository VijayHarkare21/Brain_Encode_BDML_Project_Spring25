import pickle as pkl
import numpy as np

# Load the pickle file
# with open(r'D:\Vijay\NYU\Spring_25\BDMLS\Project\dataset\ZuCo\task2-NR-2.0\pickle\task2-NR-2.0-dataset_YAC_embeds.pickle', 'rb') as f:
#     data = pkl.load(f)

# Print the data
# print(data['YAC'][0]['embeds'].shape)

data = np.load(r'D:\Vijay\NYU\Spring_25\BDMLS\Project\dataset\ZuCo\task2-NR-2.0\pickle\task2-NR-2.0-dataset_YAC_embeds.npy', allow_pickle=True)

print(data)

# np.save(r'D:\Vijay\NYU\Spring_25\BDMLS\Project\dataset\ZuCo\task2-NR-2.0\pickle\task2-NR-2.0-dataset_YAC_embeds.npy', data, allow_pickle=True)

# data2 = np.load(r'D:\Vijay\NYU\Spring_25\BDMLS\Project\dataset\ZuCo\task2-NR-2.0\pickle\task2-NR-2.0-dataset_YAC.pickle', allow_pickle=True)

# np.save(r'D:\Vijay\NYU\Spring_25\BDMLS\Project\dataset\ZuCo\task2-NR-2.0\pickle\task2-NR-2.0-dataset_YAC.npy', data2, allow_pickle=True)