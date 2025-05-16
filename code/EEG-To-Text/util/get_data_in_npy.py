#!/usr/bin/env python3
import os
import sys
import traceback
import numpy as np
import h5py
import data_loading_helpers_modified as dh
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
task       = "NR"
rootdir    = "/scratch/vjh9526/bdml_2025/project/datasets/ZuCo2/2urht/osfstorage/task1 - NR/Matlab files"
output_dir = "/scratch/vjh9526/bdml_2025/project/datasets/ZuCo2/2urht/osfstorage/task1 - NR/npy_file"
task_name  = "task1-NR-2.0"

os.makedirs(output_dir, exist_ok=True)


# -----------------------------------------------------------------------------
# WORKER FUNCTION
# -----------------------------------------------------------------------------
def process_file(filename: str):
    """
    Load one .mat file, extract all sentence+word data,
    and return (subject, list_of_sent_objs).
    On error, returns None and logs a stack trace to stderr.
    """
    try:
        # only process .mat files for our task
        if not filename.endswith(task + ".mat"):
            return None

        # parse subject (skip YMH)
        subject = filename.split("ts", 1)[1].split("_", 1)[0]
        if subject == "YMH":
            return None

        sent_list = []
        fullpath = os.path.join(rootdir, filename)
        with h5py.File(fullpath, "r") as f:
            sentence_data = f['sentenceData']
            
            # sent level eeg 
            mean_t1_objs = sentence_data['mean_t1']
            mean_t2_objs = sentence_data['mean_t2']
            mean_a1_objs = sentence_data['mean_a1']
            mean_a2_objs = sentence_data['mean_a2']
            mean_b1_objs = sentence_data['mean_b1']
            mean_b2_objs = sentence_data['mean_b2']
            mean_g1_objs = sentence_data['mean_g1']
            mean_g2_objs = sentence_data['mean_g2']
            
            rawData = sentence_data['rawData']
            contentData = sentence_data['content']
            omissionR = sentence_data['omissionRate']
            wordData = sentence_data['word']

            for idx in range(len(rawData)):
                # get sentence string
                obj_reference_content = contentData[idx][0]
                sent_string = dh.load_matlab_string(f[obj_reference_content])
                
                sent_obj = {'content': sent_string}
                sent_obj['rawData'] = np.squeeze(f[rawData[idx][0]][()])

                # get sentence level EEG
                sent_obj['sentence_level_EEG'] = {
                    'mean_t1': np.squeeze(f[mean_t1_objs[idx][0]][()]),
                    'mean_t2': np.squeeze(f[mean_t2_objs[idx][0]][()]),
                    'mean_a1': np.squeeze(f[mean_a1_objs[idx][0]][()]),
                    'mean_a2': np.squeeze(f[mean_a2_objs[idx][0]][()]),
                    'mean_b1': np.squeeze(f[mean_b1_objs[idx][0]][()]),
                    'mean_b2': np.squeeze(f[mean_b2_objs[idx][0]][()]),
                    'mean_g1': np.squeeze(f[mean_g1_objs[idx][0]][()]),
                    'mean_g2': np.squeeze(f[mean_g2_objs[idx][0]][()])
                }
                sent_obj['word'] = []

                # get word level data
                word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = \
                    dh.extract_word_level_data(f, f[wordData[idx][0]])

                if word_data == {}:
                    print(f'missing sent: subj:{subject} content:{sent_string}, append None')
                    sent_list.append(None)
                    continue
                elif len(word_tokens_all) == 0:
                    print(f'no word level features: subj:{subject} content:{sent_string}, append None')
                    sent_list.append(None)
                    continue

                for widx in range(len(word_data)):
                    data_dict = word_data[widx]
                    word_obj = {'content': data_dict['content'], 'nFixations': data_dict['nFix']}
                    if 'GD_EEG' in data_dict:
                        gd = data_dict['GD_EEG']
                        ffd = data_dict['FFD_EEG']
                        trt = data_dict['TRT_EEG']
                        assert len(gd) == len(trt) == len(ffd) == 8
                        word_obj['word_level_EEG'] = {
                            'GD': {'GD_t1': gd[0], 'GD_t2': gd[1], 'GD_a1': gd[2], 'GD_a2': gd[3], 'GD_b1': gd[4], 'GD_b2': gd[5], 'GD_g1': gd[6], 'GD_g2': gd[7]},
                            'FFD': {'FFD_t1': ffd[0], 'FFD_t2': ffd[1], 'FFD_a1': ffd[2], 'FFD_a2': ffd[3], 'FFD_b1': ffd[4], 'FFD_b2': ffd[5], 'FFD_g1': ffd[6], 'FFD_g2': ffd[7]},
                            'TRT': {'TRT_t1': trt[0], 'TRT_t2': trt[1], 'TRT_a1': trt[2], 'TRT_a2': trt[3], 'TRT_b1': trt[4], 'TRT_b2': trt[5], 'TRT_g1': trt[6], 'TRT_g2': trt[7]}
                        }
                        sent_obj['word'].append(word_obj)

                sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
                sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
                sent_obj['word_tokens_all'] = word_tokens_all
                sent_list.append(sent_obj)

        return subject, sent_list

    except Exception:
        print(f"ERROR processing {filename}:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None

# -----------------------------------------------------------------------------
# MAIN: SEQUENTIAL PROCESSING + SAVE PER SUBJECT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # list and sort all task .mat files
    files = sorted(f for f in os.listdir(rootdir) if f.endswith(task + ".mat"))

    # process sequentially with progress bar
    for filename in tqdm(files, desc="Processing subjects"):
        result = process_file(filename)
        if result is None:
            continue

        subj, sent_list = result

        # prepare output path
        outpath = os.path.join(
            output_dir,
            f"{task_name}-{subj}-dataset.npy"
        )
        # skip existing files
        if os.path.exists(outpath):
            print(f"SKIPPING existing file: {outpath}")
            continue

        # save new data
        out_dict = {subj: sent_list}
        np.save(outpath, out_dict, allow_pickle=True)
        print(f"WROTE: {outpath}")

    print("All done!")
