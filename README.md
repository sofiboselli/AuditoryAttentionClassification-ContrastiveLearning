# AuditoryAttentionClassification-ContrastiveLearning

This is the code for the final thesis titled "Auditory Attention Classification with Contrastive Learning" by Sofía Boselli and Gautam Sridhar of the master degree in Machine Learning, Systems and Control at LTH, Lund University. 

## ABSTRACT

> Auditory attention detection is crucial for understanding speech in noisy environments, a challenge known as the "cocktail party problem." This project investigates the use of electroencephalography (EEG) to identify which speaker a listener attends to. EEG's portability and real-time recording capabilities make it a promising tool for practical applications.

> We propose a novel neural network model for auditory attention detection using EEG data. The model reconstructs the attended speech envelope while simultaneously classifying attended vs. unattended speech. It incorporates a contrastive learning loss function (SigLIP), which, to our knowledge, has not been previously applied to EEG-based auditory attention detection. The model architecture combines convolutional, fully connected, and attention layers.

> Evaluated on an EEG dataset with 31 subjects, the model achieves a mean accuracy of 68\% and a mean correlation of 0.105 between the reconstructed and attended envelopes. This surpasses the baseline performance of linear methods (63\% accuracy, 0.084 correlation). These results suggest the potential of contrastive learning for improving auditory attention detection accuracy, warranting further investigation.

## About The Code

The provided code allows to make a hyperparameter search with the function `getSearchFunction` or to run only one case with the function `train_one_case`. An example of the last one is:

```
config = {
    "overlap": 0.7,
    "lr": 6e-4,
    "batch_size": 128,
    "temp": 10,
    "embedding": 8, 
    "rec_embedding": 32,
    "weight_decay": 0,
    "env_dropout": 0.4, 
    "eeg_dropout": 0.2, 
    "kernels": [30,20,40],
    "bias": 10,
}

enc, test_loader = train_one_case(config, None, False)
torch.save(enc, '/mimer/NOBACKUP/groups/naiss2023-22-692/Sofia-Gautam/HyperParameterSearch/final_python_model.pth')
```

The final report on this project can be found here
