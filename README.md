# Facial Expression Recognition
## Affective Computing - Programming Assignment
### Objective
• Task is to extract a set of prosodic correlates (i.e. suprasegmental speech parameters) and cepstral features from speech recordings. Then, an emotion recognition system is constructed to recognize happy versus sad emotional speech (a quite easy two class problem) using a simple supervised classifier training and testing structure.

• The original speech data is a set of simulated emotional speech (i.e. acted) from ten speakers speaking five different pre-segmented sentences of roughly 2-3 seconds in two different emotional states (happy and sad) totaling 100 samples. Basic prosodic features (i.e. distribution parameters derived from the prosodic correlates) are extracted using a simple voiced/unvoiced analysis of speech, pitch tracker, and energy analysis. Another set of Mel-Frequency Cepstral Coefficients (MFCC) features are also calculated for comparison.

• To produce speaker independent and content dependent emotion recognition case (i.e. while a same persons samples are not included in both training and testing sets, the same sentences are spoken in both the training and the testing sets) that could correspond to a public user interface with specific commands.

• Support Vector Machine (SVM) classifiers are trained. A random subset of 1/2 of the available speech data (i.e. half of the persons) is used to train the emotion recognition system, first using a set of simple prosodic parameter features and then a classical set of MFCC derived features. The rest of the data (the other half of the persons) is then used to evaluate the performances of the trained recognition systems.
### Task 1. Feature Extraction
1. MFCC calculations using the provided sample speech signal.

2. Extract the Intensity/Energy parameter.

3. Extract the Pitch/F0 feature.

4. Extract the Rhythm/Durations parameter.

5. Check the extracted feature.
### Task 2. Speech Emotion Classification
