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
```bash
python3 ./src/main.py -h

usage: main.py [-h] [--smooth SMOOTH] [--n_rows N_ROWS] [--n_splits N_SPLITS]
optional arguments:
  -h, --help           show this help message and exit
  --smooth SMOOTH      Value for Laplace Smoothing
  --n_rows N_ROWS      Amount of rows to read from csv file
  --n_splits N_SPLITS  K_Fold splits
```
