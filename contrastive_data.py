import numpy as np
import os
from torch.utils.data import Dataset
import torch
import scipy
from sklearn.preprocessing import minmax_scale

def findfiles(search_dir, prefix):
    matching_files = []

    for dirpath, _, filenames in os.walk(search_dir):
        for filename in filenames:
            if filename.startswith(prefix):
                full_path = os.path.join(dirpath, filename)
                matching_files.append(full_path)

    return matching_files

def window_data(data, window_length, hop):
    new_data = np.empty(((data.shape[1] - window_length) // hop,data.shape[0], window_length))
    for i in range(new_data.shape[0]):
        new_data[i, :, :] = data[:,i * hop : i * hop + window_length]
    return new_data

class EEG_Dataset(Dataset):

    def __init__(self,seconds=4,overlap=0.75,fs=64,path="EriksholmFiles/",files=[""],ear=False):
        super().__init__()

        self.seconds = seconds
        self.overlap = 1-overlap
        self.fs = fs
        self.path = path
        self.ear = ear

        self.win_len = (int(self.fs*33) - int(self.seconds*self.fs)) // int(self.seconds*self.fs*self.overlap)
        self.times = self.seconds*self.fs
        self.min = np.array([ 2.16468529e-06,  6.03260377e-06,  5.93159986e-06,  3.62664469e-06,
        2.80348336e-06,  1.42119982e-06,  6.88372887e-07,  4.84848368e-07,
        3.54443476e-07,  3.71692982e-07,  3.38777447e-07,  3.01392456e-07,
        2.67927776e-07,  1.82427235e-07,  1.42586145e-07,  1.09978849e-07,
        8.09435645e-08,  4.96393428e-08,  4.06621411e-08,  4.60418826e-08,
        4.52268590e-08,  4.04971857e-08,  4.31186742e-08,  4.47901477e-08,
        2.22082786e-08,  1.39621117e-08,  9.63034406e-09,  7.66013121e-09,
        6.81405501e-09,  8.25439855e-09,  8.64246423e-09,  6.34408143e-09,
       -4.77543935e-04])
        self.max = np.array([0.0121103 , 0.04227614, 0.08793171, 0.0764621 , 0.0801652 ,
       0.04999707, 0.02793529, 0.02279484, 0.01779334, 0.01773479,
       0.01973905, 0.01774889, 0.01933183, 0.0110212 , 0.0078137 ,
       0.00536504, 0.00304324, 0.00191434, 0.00121807, 0.00142606,
       0.00150114, 0.00150256, 0.00158835, 0.00136863, 0.00072251,
       0.00046471, 0.00032209, 0.0003018 , 0.00038911, 0.00039234,
       0.000559  , 0.00046365, 0.02558053])

        self.min = np.expand_dims(self.min,axis=1)
        self.max = np.expand_dims(self.max,axis=1)
        

        self.files = files

    def __len__(self):
        return self.win_len*len(self.files)
    
    def __getitem__(self, index):


        sub = index//(self.win_len)
        ind = int(index%(self.win_len))
        numb = int(self.files[sub].split("_")[1])
        rr = np.array([numb])
        rr = torch.Tensor(rr).to(torch.int64)

        self.data = np.load(self.files[sub])
        self.eeg = self.data['EEG']
        self.att = self.data['attended']
        self.att[:-1] = np.abs(scipy.signal.hilbert((self.att[:-1])))
        self.mas = self.data['masker']
        self.mas[:-1] = np.abs(scipy.signal.hilbert((self.mas[:-1])))
        self.loc = self.data['loc']
        win_loc = torch.Tensor(np.array([self.loc]))

        if self.ear:
            self.eeg = self.eeg[[7,8,15,16,9,14,17,23,42,43,52,53,44,51,54,60], :]
        

        win_eeg = torch.from_numpy(np.reshape(window_data(self.eeg,int(self.seconds*self.fs),int(self.seconds*self.fs*self.overlap)),(self.win_len,self.eeg.shape[0],self.times))[ind]).float()
        win_att = torch.from_numpy(np.reshape(window_data(self.att,int(self.seconds*self.fs),int(self.seconds*self.fs*self.overlap)),(self.win_len,self.att.shape[0],self.times))[ind]).float()
        win_mas = torch.from_numpy(np.reshape(window_data(self.mas,int(self.seconds*self.fs),int(self.seconds*self.fs*self.overlap)),(self.win_len,self.mas.shape[0],self.times))[ind]).float()

        self.eeg = self.eeg[:,:]


        win_eeg = torch.nn.functional.normalize(win_eeg,p=2,dim=(1))
        win_att = torch.nn.functional.normalize(win_att,p=2,dim=(1))
        win_mas = torch.nn.functional.normalize(win_mas,p=2,dim=(1))
        

        
        
        
        return win_eeg,win_att,win_mas,win_loc,rr


            
            
