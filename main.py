#!/usr/bin/env python3
"""DESCRIPTION OF THE PROGRAM

(C) 2021 Clément SEIJIDO
Released under GNU General Public License v3.0 (GNU GPLv3)
e-mail clement@seijido.fr
"""

import string
import re
import pathlib

import nltk
import pandas


TRAINING_DIRECTORY = pathlib.Path('20news-bydate/20news-bydate-train/')
TESTING_DIRECTORY = pathlib.Path('20news-bydate/20news-bydate-test/')


def training():
    """For all categories
         For all files
           remove stop words
           lemmatisation
           vectorisation
    """

    categories = [p.name for p in TRAINING_DIRECTORY.iterdir()]
    categories.sort()

    training_set = pandas.DataFrame(columns=['label', 'text'])
    vocabulary = set()  # ignore duplicates

    for category in categories:
        print(f'Collecting {category}...')
        category_path = TRAINING_DIRECTORY / category
        files_path = list(category_path.iterdir())
        files_path.sort()
        for message_path in files_path:
            message = read_message(message_path)
            word_list = clean_text(message)
            training_set = training_set.append(
                {'label': category, 'text': word_list},
                ignore_index=True
            )
            vocabulary.update(word_list)

    word_counts_per_message = {
        word: [0] * len(training_set['text']) for word in vocabulary
    }

    for index, message in enumerate(training_set['text']):
        for word in message:
            word_counts_per_message[word][index] += 1

    word_counts = pandas.DataFrame(word_counts_per_message)
    training_set = pandas.concat([training_set, word_counts], axis=1)
    print(training_set.head())
    print('Vocabulary:', len(vocabulary))


def read_message(message_path, encoding='utf-8'):
    wrong_lines_start = [
        'From:',
        'Subject:',
        'X-Xxmessage-Id:',
        'X-Xxdate:',
        'Summary:',
        'Keywords:',
        'Expires:',
        'Distribution:',
        'Organization:',
        'Supersedes:',
        'News-Software:',
        'X-Newsreader:',
        'X-Useragent:',
        'Lines:',
        'Archive-name:',
        'Last-modified:',
        'Version:',
        'NNTP-Posting-Host:',
        'Nntp-Posting-Host:',
        'Reply-To:',
        'Disclaimer:',
        'In article',
        '>',
        '|>',
        ':',
    ]

    wrong_lines_end = [
        'writes:',
        'wrote:',
    ]

    message = ''
    try:
        with open(message_path, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                good_line = True
                for start in wrong_lines_start:
                    if line.startswith(start):
                        good_line = False
                for end in wrong_lines_end:
                    if line.endswith(end):
                        good_line = False

                if good_line:
                    message += line + '\n'

        return message

    except UnicodeDecodeError:
        return read_message(message_path, encoding='iso-8859-1')


def clean_text(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    word_list = []  # create a list of words
    for word in re.split(r'\W+', text):
        word = word.lower()
        if word and word not in nltk.corpus.stopwords.words('english') \
                and word not in string.punctuation:
            word_list.append(lemmatizer.lemmatize(word))

    return word_list


if __name__ == '__main__':
    training()
