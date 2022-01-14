# Naive Bayes text analysis
[![Python Version: 3.10](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge&logo=python)](https://github.com/Calibanda/naive_bayes_text_analysis/)
[![Open Source? Yes!](https://img.shields.io/badge/Open%20Source%3F-Yes!-green?style=for-the-badge&logo=appveyor)](https://github.com/Calibanda/naive_bayes_text_analysis/)
[![License: GNU GPLv3](https://img.shields.io/github/license/Calibanda/naive_bayes_text_analysis?style=for-the-badge)](https://github.com/Calibanda/naive_bayes_text_analysis/blob/main/LICENSE)

Naive Bayes classifier on the 20 Newsgroups data set.

## Installation

### Clone the repository

Clone this repository in your personal directory with the command:

```
git clone https://github.com/Calibanda/naive_bayes_text_analysis.git
```

### Create a new virtual environment

On Linux or MacOS

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python3 -m pip install --upgrade pip
```

On Windows

```shell
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

*For more information about virtual environments see the [official documentation](https://docs.python.org/3/library/venv.html).*

### Install needed packages

Install needed packages with:

```bash
pip install -r requirements.txt
```

### Install nltk_data

Install nltk_data with the command:

```bash
python -m nltk.downloader -d .venv/nltk_data/ stopwords wordnet omw-1.4
```


## Running the program

Execute the following command to start the program:

```bash
python main.py
```
