# Word Alignment -- IBM Model 1

Implementation of  the EM algorithm applied to IBM Model 1 for machine translation

## Setup

All the project dependencies are inside the `requirements.txt`. Install them using pip:
```
pip install -r requirements.txt
```

## Usage

```
usage: main.py [-h] [--iterations ITERATIONS] [--sentences SENTENCES] [--top TOP] [--punctuation_allowed] [--to_lower]

options:
  -h, --help            show this help message and exit
  --iterations ITERATIONS
                        Number of iterations to perform EM algorithm
  --sentences SENTENCES
                        Number of sentences from the eng-cz corpus
  --top TOP             Returns best n translations of the current CZ word
  --punctuation_allowed
                        Remove punctuations in the preprocessing
  --to_lower            Lowers the casing of words in the preprocessing

```

## Default Parameters

* ```--iterations=5```
* ```--sentences=2501```
* ```--top=3```
* ```--punctuation_allowed=True```
* ```--to_lower=False```


## How to run

You can run the script with default parameters from the command-line using
```
make

```
