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




class GetTrainDataset(Dataset):
    # Initialize your data, download, etc. 
    def __init__(self, dir_list, label_csv='./train_answer.csv', class_list=[], label_image=True): 
        self.dir_list = np.sort(dir_list)
        self.len = self.dir_list.shape[0] 
        self.label_image = label_image
        label_df = pd.read_csv(label_csv)
        self.label_df = pd.read_csv('./train_answer.csv')
        label = []
        if len(class_list) > 0:
            for label_name in class_list:
                label.append(np.int32(label_df[label_name].values!=0)) # label name에서 값이 있는 부분만 1로지정
        else:
            for label_name in label_df.columns[1:]:
                label.append(np.int32(label_df[label_name].values!=0)) # label name에서 값이 있는 부분만 1로지정
        self.label = np.array(label) # (n_class, num_data)
        
        
    def __getitem__(self, index):
        # get item이 호출될 때, directory 에서 파일 로드해서 melspectrogram 연산
        fs, signal = wavfile.read(self.dir_list[index])
        mel_result = self.get_melspectrogram(signal, fs=fs, nfft=320, normalize=True)
        w, h = mel_result.shape
        # pytorch input shape [channels, height, width]
        self.x_data = torch.from_numpy(mel_result[np.newaxis, :, :])  
        
        if np.sum(self.label[:, index]) == 0:
            each_label = np.concatenate([[1], self.label[:, index]])
        else:
            each_label = np.concatenate([[0], self.label[:, index]])
        if self.label_image:
            self.y_data = torch.from_numpy(np.tile(each_label[:, np.newaxis, np.newaxis], [1, 1, w, h])[0])
            return self.x_data, self.y_data 
        else:
#             choice_one_label = np.zeros(each_label.shape)
#             num_label = len(np.where(each_label==1)[0])
#             loc_label = np.where(each_label==1)[0]
#             choice_one_label[loc_label[np.random.randint(num_label)]] = 1
            self.y_data = each_label
            return self.x_data, self.y_data
        
        
    
    def __len__(self):
        return self.len 
    
    # melSpectrogram 을 얻기위한 함수
    def get_melspectrogram(self, signal, fs, nfft, normalize):
        hop_length = int(fs/nfft)
        n_mles = nfft/2
        mel_result = librosa.feature.melspectrogram(y=np.float32(signal), sr=fs, n_mels=n_mles, hop_length=hop_length) # melspectrogram 
        result_db = librosa.power_to_db(mel_result, ref=np.max) # 데시벨로 데이터 범위 한정 (로그를씌웟나 그랫던거같음)
#         stft_result = librosa.stft(np.float32(signal), n_fft=nfft, hop_length=hop_length)
#         result_db = librosa.amplitude_to_db(np.abs(stft_result))
        if normalize:
            _min = np.min(result_db)
            _max = np.max(result_db)
            result_db = (result_db-_min)/(_max-_min)
        #print (mel_result_db.shape)
        return result_db


    
    
    