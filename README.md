# Audio-Classification

Projet of audio classification using Pytorch and torchaudio. The goal is to create a sound classifier for the SpeechCommands dataset from torchaudio, which is a dataset of 35 commands spoken by different people, and made up of over 105 800 recordings. All audio files are about 1 second long (and so about 16000 time frames long).

In this project, three models have been trained using the raw spectograms, the MFCC features and the MelSpectogram features. The MFCCs provide the best results with a test accuracy of around 85%.

