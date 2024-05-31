import torch
import pytorch_lightning as pl
from audtorch.metrics import PearsonR
from contrastive_data import EEG_Dataset
from torchmetrics.wrappers import Running
from torch.utils.data import DataLoader,TensorDataset,Sampler
import random
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,ModelSummary
import os
from sklearn.model_selection import train_test_split
import numpy as np
from torch import nn
from braindecode.augmentation import AugmentedDataLoader, FTSurrogate, SmoothTimeMask, FrequencyShift, ChannelsDropout,ChannelsShuffle,GaussianNoise
import torch.nn.functional as F
from braindecode.augmentation import AugmentedDataLoader, FTSurrogate, SmoothTimeMask, FrequencyShift, ChannelsDropout, GaussianNoise, SignFlip, BandstopFilter, ChannelsSymmetry, ChannelsShuffle, SensorsRotation 

from ray.train.lightning import (RayDDPStrategy, RayLightningEnvironment, RayTrainReportCallback, prepare_trainer)
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

torch.set_float32_matmul_precision('medium')

def findfiles(search_dir, subjectstring):
    matching_files = []

    for dirpath, _, filenames in os.walk(search_dir):
        for filename in filenames:
            if subjectstring in filename:
                full_path = os.path.join(dirpath, filename)
                matching_files.append(full_path)

    return matching_files

class Pearsonr(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,P,O):
        (n,c,d) = O.shape      # n traces of t samples

        DO = O - (torch.einsum("ncd->n", O) / (c*d)).reshape(-1,1,1) # compute O - mean(O)
        DP = P - (torch.einsum("ncd->n", P) / (c*d)).reshape(-1,1,1) # compute P - mean(P)

        cov = torch.einsum("ncd,mcd->nm", DP, DO)
        varP = torch.einsum("nm...,nm...->n", DP, DP)
        varO = torch.einsum("nt...,nt...->n", DO, DO)
        tmp = torch.einsum("m,t->mt", varP, varO)

        return cov / torch.sqrt(tmp)

class CLIP(torch.nn.Module):
    def __init__(self,scale,temp,bias=-10,acc=False,gamma=1):
        super().__init__()
        self.scale = scale
        self.temp = temp
        self.pr = Pearsonr()
        self.acc = acc
        self.bias = bias
    
    def forward(self,eemb,aamb):
   
        if self.scale:
            eemb =(eemb - eemb.mean(dim=(1,2),keepdim=True))/eemb.std(dim=(1,2),keepdim=True)
            aamb =(aamb - aamb.mean(dim=(1,2),keepdim=True))/aamb.std(dim=(1,2),keepdim=True)

        eemb = nn.functional.normalize(eemb,p=2,dim=(1,2))
        aamb = nn.functional.normalize(aamb,p=2,dim=(1,2))
    
        logits = torch.einsum('btc,rtc->br',eemb,aamb)*torch.exp(self.temp) + self.bias
        labels = 2*torch.eye(eemb.shape[0],device=eemb.device) - torch.ones((eemb.shape[0],eemb.shape[0]),device=eemb.device)

        if self.acc == True:
            return logits

        return -torch.sum(F.logsigmoid(labels * logits)) / (eemb.shape[0])

class EnvConv(nn.Module):

    def __init__(self, embedding, dropout, kernels):
        super().__init__()
        self.conv = nn.Conv1d(33,embedding,kernel_size=3,padding="same")
        self.kernels = kernels
        #self.kernels = [31,7,7,7,7,3,5,19]
        self.conv2 = nn.ModuleList([nn.Conv1d(embedding,embedding,kernel_size=k,padding="same") for k in self.kernels])
        self.act = nn.ModuleList([nn.GELU() for k in self.kernels])
        self.drop = nn.ModuleList([nn.Dropout(dropout) for k in self.kernels])
        self.reconv = nn.Conv1d(embedding,embedding,kernel_size=1)
        self.bnorm = nn.ModuleList([nn.InstanceNorm1d(embedding) for k in self.kernels])
        self.re = nn.ReLU()
        self.mh = nn.ModuleList([nn.MultiheadAttention(embedding,2,batch_first=True) for k in self.kernels])
        
    def forward(self,x):
        x = self.re(x)
        x = self.conv(x)
        for conv,act,drop,bnorm,mh in zip(self.conv2,self.act,self.drop,self.bnorm, self.mh):
            out = conv(x)

            out = out.transpose(1,2) 
            ch,_ = mh(out,out,out) 
            out = out.transpose(1,2) 
            ch = ch.transpose(1,2)
            out = out+ch 
            
            out = bnorm(out)
            x = drop(out)
        
        out = self.reconv(x)
        
        return out

class SubjectLayers(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x, subjects):
        _, C, D = self.weights.shape
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)

class EEGConv(nn.Module):

    def __init__(self, embedding, dropout, kernels, eeg_channels):
        super().__init__()
        self.kernels = kernels
        self.k2d = [3,2,3]
        self.conv = nn.ModuleList([nn.Conv1d(embedding,embedding,kernel_size=k,padding="same") for k in self.kernels])
        self.act = nn.ModuleList([nn.GELU() for k in self.kernels])
        self.drop = nn.ModuleList([nn.Dropout(dropout) for k in self.kernels]) 
        self.co = nn.Conv1d(eeg_channels,embedding,kernel_size=3,padding="same")
        self.lnorm = nn.ModuleList([nn.InstanceNorm1d(embedding) for k in self.kernels])
        self.convolve = nn.Conv1d(embedding,embedding,kernel_size=1) 
        self.sublayer = SubjectLayers(eeg_channels,eeg_channels,33)
        self.mh = nn.ModuleList([nn.MultiheadAttention(embedding,2,batch_first=True) for k in self.kernels])

       
    def forward(self,x,subj):
        
        sub = self.sublayer(x,subj)
        x = x+sub
        x = self.co(x)
        for conv,act,drop,bnorm,mh in zip(self.conv,self.act,self.drop,self.lnorm, self.mh):
            out = conv(x)

            out = out.transpose(1,2) 
            ch,_ = mh(out,out,out) 
            out = out.transpose(1,2) 
            ch = ch.transpose(1,2) 
            out = out+ch 
            
            out = drop(out)
            out = bnorm(out)
            x = x+out

        
        out = self.convolve(x)

        return out

class recon(nn.Module):

    def __init__(self, rec_embedding, embedding, kernels):
        super().__init__()
        self.kernels = kernels
        self.rekern = self.kernels[::-1]
        self.conv = nn.ModuleList([nn.Conv1d(rec_embedding,rec_embedding,kernel_size=k,padding="same") for k in self.rekern])
        self.act = nn.ModuleList([nn.GELU() for k in self.kernels])
        self.bnorm = nn.ModuleList([nn.InstanceNorm1d(rec_embedding) for k in self.kernels])
        self.drop = nn.ModuleList([nn.Dropout(0.2) for k in self.kernels]) 
        self.co = nn.Conv1d(embedding,rec_embedding,kernel_size=13,padding="same")
        self.con = nn.Conv1d(rec_embedding,1,kernel_size=1)
        self.act2 = nn.GELU()
        self.mh = nn.ModuleList([nn.MultiheadAttention(rec_embedding,2,batch_first=True) for k in self.kernels])

    def forward(self,x):
        x = self.co(x)
        
        for conv,act,drop,bnorm,mh in zip(self.conv,self.act,self.drop,self.bnorm,self.mh):
            out = conv(x)
            
            out = out.transpose(1,2) 
            ch,_ = mh(out,out,out) 
            out = out.transpose(1,2) 
            ch = ch.transpose(1,2) 
            out = out+ch

            out = drop(out)
            out = bnorm(out)

            x = out
        x = self.con(x)
        x = self.act2(x)
        return x.squeeze(1)


class Encode(pl.LightningModule):
    def __init__(self, lr, embedding_size, rec_embedding, temp, weight_decay, env_drop, eeg_drop, kernels, bias, eeg_channels):
        super().__init__()
        self._initialize_parameters(temp, bias)
        self._initialize_modules(embedding_size, eeg_drop, env_drop, kernels, rec_embedding, eeg_channels)
        self._initialize_metrics()
        self._initialize_hyperparameters(lr,  weight_decay)

    def _initialize_parameters(self, temp, bias):
        self.temp = nn.Parameter(torch.ones([]) * np.log(temp))
        self.bias = nn.Parameter(torch.ones([]) * -bias)
        
    def _initialize_modules(self, embedding_size, eeg_drop, env_drop, kernels, rec_embedding, eeg_channels):
        self.encoder = EEGConv(embedding_size, eeg_drop, kernels, eeg_channels)
        self.audproj = EnvConv(embedding_size, env_drop, kernels)
        self.reconstruct = recon(rec_embedding, embedding_size, kernels)
        self.loss_fn = CLIP(scale=True, temp=self.temp, bias=self.bias)
        self.afn = CLIP(scale=True, temp=self.temp, acc=True, bias=self.bias)
        self.los = PearsonR()

    def _initialize_metrics(self):
        self.training_losses = []
        self.validation_losses = []
        self.tr_pearson = []
        self.val_pearson = []
        self.tr_acc = []
        self.val_acc = []
        self.val_acc_plot = []
        self.val_loss_plot = []
        
    def _initialize_hyperparameters(self, lr,  weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay
        
    def forward(self, eeg):
        return self.encoder(eeg)

    def training_step(self, batch, batch_idx):
        eeg, att, mas, _, subj = batch
        eeg_emb = self.encoder(eeg,subj)
        re = self.reconstruct(eeg_emb)

        attf = self.audproj(att)
        masf = self.audproj(mas)

        env = att[:,-1]
        menv = mas[:,-1]

        loss = self.loss_fn(eeg_emb, attf) - self.los(re, env)
        pearson = self.los(re, env)
        
        self._update_metrics(loss, pearson, self.tr_pearson, self.training_losses, "running_train_loss", "pearsonr")
        
        acc = self._calculate_accuracy(eeg_emb, attf, masf)
        self.tr_acc.append(acc.item())
        self.log("train_accuracy", np.mean(self.tr_acc), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        eeg, att, mas, _, subj = batch
        eeg_emb = self.encoder(eeg,subj)
        re = self.reconstruct(eeg_emb)

        attf = self.audproj(att)
        masf = self.audproj(mas)

        env = att[:,-1]
        menv = mas[:,-1]

        loss = self.loss_fn(eeg_emb, attf) - self.los(re, env)
        pearson = self.los(re, env)
        
        self._update_metrics(loss, pearson, self.val_pearson, self.validation_losses, "val_loss", "val_pearsonr")

        acc = self._calculate_accuracy(eeg_emb, attf, masf)
        self.val_acc.append(acc.item())
        self.log("val_accuracy", np.mean(self.val_acc), prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        self._reset_metrics()

    def on_validation_epoch_end(self):
        self.val_acc_plot.append(np.mean(self.val_acc))
        self.val_loss_plot.append(np.mean(self.validation_losses))
        self._reset_metrics(validation=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def _update_metrics(self, loss, pearson, pearson_list, losses_list, loss_key, pearson_key):
        pearson_list.append(pearson.item())
        losses_list.append(loss.item())
        
        running_avg = np.mean(losses_list)
        self.log(loss_key, running_avg, prog_bar=True)
        
        running_pearson = np.mean(pearson_list)
        self.log(pearson_key, running_pearson, prog_bar=True)

    def _calculate_accuracy(self, eeg_emb, attf, masf):
        m1 = torch.diagonal(self.afn(eeg_emb, attf))
        m2 = torch.diagonal(self.afn(eeg_emb, masf))
        
        acc = torch.sum(m1 > m2) / m1.shape[0]
        return acc
    
    def _reset_metrics(self, validation=False):
        if validation:
            self.validation_losses.clear()
            self.val_acc.clear()
            self.val_pearson.clear()
        else:
            self.training_losses.clear()
            self.tr_acc.clear()
            self.tr_pearson.clear()

class ResamplingSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.indices = list(range(len(data_source)))
        self.original_indices = list(range(len(data_source)))

    def __iter__(self):
        # Shuffle indices to get a random order
        random.shuffle(self.indices)
        # Initialize an empty list to store sampled indices
        sampled_indices = []
        firstatt = []
        batch = []
        
        # Iterate over shuffled indices
        for index in self.indices:
            # Get the attributes from the dataset for the current index
            _, att, _, _, _ = self.data_source[index]
            
            
            # Check if the current attribute 'att' exists in the sampled_indices
            if att[0][0] not in firstatt:
                firstatt.append(att[0][0])
                sampled_indices.append(index)
            else:
                # Resample if a duplicate 'att' is found
                while True:
                    resampled_index = random.choice(self.indices)
                    _, resampled_att, _, _, _ = self.data_source[resampled_index]
                    if resampled_att[0][0] not in firstatt:
                        firstatt.append(resampled_att[0][0])
                        sampled_indices.append(resampled_index)
                        break
            yield sampled_indices

    def __len__(self):
        return len(self.data_source)

def getLoaders(batch_size, overlap, transform=None, ear=False):
    dir = "/mimer/NOBACKUP/groups/naiss2023-22-692/Sofia-Gautam/Trypls"
    
    train = {}
    val = {}
    test = {}
    
    for i in range(1,33):
        if i ==14:
            continue
        substr = "subject_" + str(i) + "_"
    
        subjectlist = findfiles(dir,substr)
        spfiles,testfiles = train_test_split(subjectlist,test_size=0.1,random_state=42)
        trainfiles,valfiles = train_test_split(spfiles,test_size=0.12,random_state=42)
    
        train[substr] = trainfiles
        val[substr] = valfiles
        test[substr] = testfiles
        
    trains = sum(train.values(),[])
    vals = sum(val.values(),[])
    tests = sum(test.values(),[])
    
    random.shuffle(trains)
    random.shuffle(vals)
    random.shuffle(tests)

    tr_dataset = EEG_Dataset(files=trains,seconds=3,overlap=overlap,fs=64, ear=ear)
    val_dataset = EEG_Dataset(files=vals,seconds=3,overlap=0,fs=64, ear=ear)
    test_dataset = EEG_Dataset(files=tests,seconds=3,overlap=0,fs=64, ear=ear)
    
    rs = ResamplingSampler(tr_dataset)
    bs = torch.utils.data.RandomSampler(rs)

    if transform != None:
        tr_loader = AugmentedDataLoader(tr_dataset,transforms=transform,num_workers=4,persistent_workers=True,batch_size=batch_size,sampler = bs)
    else:
        tr_loader = DataLoader(tr_dataset,num_workers=4,persistent_workers=True,batch_size=batch_size,sampler = bs)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,num_workers=4,persistent_workers=True)
    test_loader = DataLoader(test_dataset,batch_size=1,num_workers=4,persistent_workers=True)

    return tr_loader, val_loader, test_loader

def train_func(config):

    tr_loader, val_loader, test_loader = getLoaders(config["batch_size"], config["overlap"], None, False)
    
    enc = Encode(config["lr"],config["embedding"], config["rec_embedding"],config["temp"], config["weight_decay"], config["env_dropout"], config["eeg_dropout"], config["kernels"], config["bias"])

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath='/mimer/NOBACKUP/groups/naiss2023-22-692/Sofia-Gautam/HyperParameterSearch',filename='pretrained_weights',auto_insert_metric_name=False,verbose=True,enable_version_counter=False)
    early_stopping = EarlyStopping('val_loss',patience=10)
    ms = ModelSummary(max_depth=2)

    trainer = pl.Trainer(
        max_epochs=300,
        accelerator='gpu',
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback(),checkpoint_callback,ms,early_stopping],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(enc, tr_loader,val_dataloaders=val_loader)

def predict(loader, model):
    accuracy = []
    correlation = []
    model.eval()
    for batch in loader:
        eeg, att, mas, _, subj = batch
        eeg_emb = model.encoder(eeg,subj)
        re = model.reconstruct(eeg_emb)
    
        attf = model.audproj(att)
        masf = model.audproj(mas)

        env = att[:,-1]
        menv = mas[:,-1]
    
        loss = model.loss_fn(eeg_emb, attf) - model.los(re, env)
        pearson = model.los(re, env)
            
        acc = model._calculate_accuracy(eeg_emb, attf, masf)
        accuracy.append(acc.item())
        correlation.append(pearson.item())
    return np.mean(accuracy), np.mean(correlation)


def train_one_case(config, transform, ear):
    if ear:
        eeg_channels = 16
    else:
        eeg_channels = 64
    tr_loader, val_loader, test_loader = getLoaders(config["batch_size"], config["overlap"], transform, ear)
    
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath='/mimer/NOBACKUP/groups/naiss2023-22-692/Sofia-Gautam/HyperParameterSearch',filename='pretrained_weights',auto_insert_metric_name=False,verbose=True,enable_version_counter=False)
    early_stopping = EarlyStopping('val_loss',patience=10)
    ms = ModelSummary(max_depth=2)

    enc = Encode(config["lr"],config["embedding"], config["rec_embedding"],config["temp"], config["weight_decay"], config["env_dropout"], config["eeg_dropout"], config["kernels"], config["bias"], eeg_channels)

    trainer = pl.Trainer(max_epochs=300,accelerator='gpu',callbacks=[checkpoint_callback,ms,early_stopping])
    trainer.fit(enc, tr_loader,val_dataloaders=val_loader)

    print("TESTING:")
    acc, pear = predict(val_loader, enc)
    print("ON VAL DATA: ACCURACY: ", acc, " PEARSON: ", pear)
    acc, pear = predict(test_loader, enc)
    print("ON TEST DATA: ACCURACY: ", acc, " PEARSON: ", pear)
    
    return enc, test_loader


def getSearchFunction(epochs, samples, search_space):
    
    num_epochs = epochs
    num_samples = samples
    
    scaling_config = ScalingConfig(
        num_workers=3, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
    )
    
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_accuracy",
            checkpoint_score_order="max",
        ),
    )
    
    ray_trainer = TorchTrainer(train_func, scaling_config=scaling_config, run_config=run_config)

    def tune_encoder_asha():
        scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)
    
        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric="val_accuracy",
                mode="max",
                num_samples=num_samples,
                scheduler=scheduler,
            ),
        )
        return tuner.fit()

    return tune_encoder_asha


if __name__ == "__main__":

    search_space = {
        "overlap": tune.choice([0.5, 0.6, 0.7]),
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([64, 128]),
        "temp": tune.choice([5, 10]),
        "embedding": tune.choice([8, 10, 16]),
        "rec_embedding": tune.choice([30, 32, 34, 28]),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "env_dropout": tune.choice([0.2, 0.3, 0.4, 0.5]),
        "eeg_dropout": tune.choice([0.2, 0.3, 0.4, 0.5]),
        "kernels": tune.choice([[20,10,20], [3,5,9], [5,17,32], [5,5,5,4],[10,20,30],[30,20,40], [4, 10], [1,5,10],[7,9,11,9,7], [13,48,29], [50,50,50], [16,27,13], [5,30,15], [8,3,3,8], [10, 13, 15, 13, 10]]),
        "bias": tune.choice([20, 10, 5]),
    }
    
    num_epochs = 100
    num_samples = 50
    searcher = getSearchFunction(num_epochs,num_samples, search_space)
    results = searcher()

    results.get_best_result(metric="val_accuracy", mode="max")

