from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import librosa

from scipy.io import wavfile
import glob
import tqdm



################
#### config ####
################
class DictToObject(object):

    def __init__(self, dictionary):
        def _traverse(key, element):
            if isinstance(element, dict):
                return key, DictToObject(element)
            else:
                return key, element

        objd = dict(_traverse(k, v) for k, v in dictionary.items())
        self.__dict__.update(objd)


#######################
#### for data load ####
#######################
def get_melspectrogram(signal, fs, nfft):
    hop_length = int(fs/nfft)
    n_mles = nfft/2
    mel_result = librosa.feature.melspectrogram(y=np.float32(signal), sr=fs, n_mels=n_mles, hop_length=hop_length) # melspectrogram 
    mel_result_db = librosa.power_to_db(mel_result, ref=np.max) # 데시벨로 데이터 범위 한정 (로그를씌웟나 그랫던거같음)
    
    #print (mel_result_db.shape)
    return mel_result_db


class GetTrainDataset(Dataset):
    # Initialize your data, download, etc. 
    def __init__(self, dir_list): 
        self.dir_list = np.sort(dir_list)
        self.len = self.dir_list.shape[0] 
        label_df = pd.read_csv('./train_answer.csv')
        self.label_bed = np.int32(label_df.bed.values!=0) # label bed 인 부분만 꺼내서 0이 아니면 1로지정
        
        
    def __getitem__(self, index):
        # get item이 호출될 때, directory 에서 파일 로드해서 melspectrogram 연산
        fs, signal = wavfile.read(self.dir_list[index])
        mel_result = get_melspectrogram(signal, fs=fs, nfft=160)
        w, h = mel_result.shape
        # pytorch input shape [channels, height, width]
        self.x_data = torch.from_numpy(mel_result[np.newaxis, :, :])  
        self.y_data = torch.from_numpy(np.tile(self.label_bed[index, np.newaxis, np.newaxis], [1, w, h])[0]) 
        
        return self.x_data, self.y_data # 현재의문사항 앞에 batch_size없어도되나?
    
    def __len__(self):
        return self.len 
    
    # melSpectrogram 을 얻기위한 함수
    def get_melspectrogram(self, signal, fs, nfft):
        hop_length = int(fs/nfft)
        n_mles = nfft/2
        mel_result = librosa.feature.melspectrogram(y=np.float32(signal), sr=fs, n_mels=n_mles, hop_length=hop_length) # melspectrogram 
        mel_result_db = librosa.power_to_db(mel_result, ref=np.max) # 데시벨로 데이터 범위 한정 (로그를씌웟나 그랫던거같음)

        #print (mel_result_db.shape)
        return mel_result_db
    
    