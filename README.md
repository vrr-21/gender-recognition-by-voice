# gender-recognition-by-voice


Using k-NN classifier with neighbors=1, to distinguish between whether the voice is that of a male or female.

Dataset available on Kaggle --> https://www.kaggle.com/primaryobjects/voicegender

Contents of repo:
* genderByVoice.py is the file which implements the classifier.
* voice.csv is the dataset.

Only the following 6 features were considered:
* meanfreq
* sd
* centroid
* meanfun
* IQR
* median

The dataset contained 3168 records, 2376 of which were used as training set and the rest as test set. The splitting of the dataset was done by `train_test_split()` from the `sklearn.model_classification` class.  
Using the k-NN classifier, 97.3485% accuracy was obtained.

**NOTE:**

The python file uses Pandas, Numpy, and Scikit-Learn, hence it is imperative to get them and their respective dependencies before using this file.
