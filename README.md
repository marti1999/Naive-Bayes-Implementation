This program is a Naive Bayes Classifier implementation using python (along with Numpy and Math libraries). Its goal is to tell whether a tweet is positive or negative. 
The Naive-Bayes class inherits from sklearn BaseEstimator and so it can be used with multiple sklearn modules like cross_validate.  
It is created with the aim to resemble models from machine learning libraries. Therefore, it can be used as one would expect by first creating the model, followed by calling fit and predict functions.    



Install dependencies
```bash
python3 -m pip install -r requirements.txt
```

Run program
```bash
python3 ./src/main.py
```

Run program with optional parameters
```bash
python3 ./src/main.py --smooth 0.1 --n_rows 10000 --n_splits 4
```

Parameters information
```
python3 ./src/main.py -h

usage: main.py [-h] [--smooth SMOOTH] [--n_rows N_ROWS] [--n_splits N_SPLITS]
optional arguments:
  -h, --help           show this help message and exit
  --smooth SMOOTH      Value for Laplace Smoothing
  --n_rows N_ROWS      Amount of rows to read from csv file
  --n_splits N_SPLITS  K_Fold splits
```
