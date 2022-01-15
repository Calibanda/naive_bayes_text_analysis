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

___

## About the 20 Newsgroups dataset

This dataset comes from: [http://qwone.com/~jason/20Newsgroups/](http://qwone.com/~jason/20Newsgroups/)

> The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. To the best of my knowledge, it was originally collected by Ken Lang, probably for his *Newsweeder: Learning to filter netnews* paper, though he does not explicitly mention this collection. The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering.

> The data is organized into 20 different newsgroups, each corresponding to a different topic. Some of the newsgroups are very closely related to each other (e.g. comp.sys.ibm.pc.hardware / comp.sys.mac.hardware), while others are highly unrelated (e.g misc.forsale / soc.religion.christian).

> [The set "bydate"] is sorted by date into training(60%) and test(40%) sets, does not include cross-posts (duplicates) and does not include newsgroup-identifying headers (Xref, Newsgroups, Path, Followup-To, Date).
