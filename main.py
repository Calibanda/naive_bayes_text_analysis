#!/usr/bin/env python3
"""DESCRIPTION OF THE PROGRAM

(C) 2021 Clément SEIJIDO
Released under GNU General Public License v3.0 (GNU GPLv3)
e-mail clement@seijido.fr
"""

import logging
import math
import re
import pathlib
import collections

import nltk
import pandas

pandas.set_option('display.max_columns', None)  # df.head() display all columns

# Data from http://qwone.com/~jason/20Newsgroups/
# Each subdirectory in the bundle represents a newsgroup;
# each file in a subdirectory is the text of some newsgroup document that
# was posted to that newsgroup.
# "bydate" is sorted by date into training(60%) and test(40%) sets,
# does not include cross-posts (duplicates) and does not include
# newsgroup-identifying headers (Xref, Newsgroups, Path, Followup-To, Date).

TRAINING_DIRECTORY = pathlib.Path('20news-bydate/20news-bydate-train/')
TESTING_DIRECTORY = pathlib.Path('20news-bydate/20news-bydate-test/')


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
        '--',
        '* ',
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
        if word and word.isalpha() \
                and word not in nltk.corpus.stopwords.words('english') \
                and len(word) > 1:
            lemmatized_word = lemmatizer.lemmatize(word)
            if len(lemmatized_word) > 1:
                word_list.append(lemmatized_word)

    return word_list


def training() -> pandas.DataFrame:
    """For all categories
         For all files
           remove stop words
           lemmatisation
           vectorisation
           probabilities
    """

    try:
        with open('trained_model.csv', 'r') as f:
            model = pandas.read_csv(f)
            model.set_index('words', inplace=True)

        logging.debug(f'{model.shape=}')
        logging.debug(model)

        return model

    except FileNotFoundError:
        pass

    categories = [p.name for p in TRAINING_DIRECTORY.iterdir()]
    categories.sort()

    model = pandas.DataFrame()
    n_words = {}

    for category in categories:
        logging.info(f'Collecting {category}...')
        category_path = TRAINING_DIRECTORY / category
        files_path = list(category_path.iterdir())
        files_path.sort()

        category_word_list = []
        for message_path in files_path:
            message = read_message(message_path)
            category_word_list += clean_text(message)

        category_counter = collections.Counter(category_word_list)
        model = pandas.concat(
            [model, pandas.Series(category_counter).rename(category)],
            axis=1  # Add as column
        )
        n_words[category] = len(category_word_list)

    model.sort_index(inplace=True)
    model.index.name = 'words'

    # apply Laplace smoothing and compute probabilities
    model.fillna(0, inplace=True)  # replace NaN by 1
    model += 1
    for category in categories:
        model[category] = model[category] / n_words[category]

    logging.debug(f'{model.shape=}')
    logging.debug(model)

    # save model for future use
    logging.info(f'Save model on disk.')
    with open('trained_model.csv', 'w') as f:
        f.write(model.to_csv())

    return model


def classify_message(message, trained_model: pandas.DataFrame) -> list:
    if isinstance(message, str):
        message = clean_text(message)

    occurrences = collections.Counter(message)

    probabilities = {category: 0 for category in trained_model.columns}

    for category in trained_model.columns:
        for word, count in occurrences.items():
            try:
                probabilities[category] += math.log10(
                    trained_model[category].loc[word]
                ) * count
            except KeyError:  # word is not in training model
                pass

    probabilities = sorted(  # sort by probability
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return probabilities


def testing(trained_model: pandas.DataFrame) -> dict:
    categories = [p.name for p in TESTING_DIRECTORY.iterdir()]
    categories.sort()

    quality = {'correct': 0, 'total': 0}

    for category in categories:
        logging.info(f'Testing with {category}...')
        category_path = TESTING_DIRECTORY / category
        files_path = list(category_path.iterdir())
        files_path.sort()

        for message_path in files_path:
            message = read_message(message_path)
            classification = classify_message(message, trained_model)
            if classification[0][0] == category:
                quality['correct'] += 1
            quality['total'] += 1

    return quality


def main():
    trained_model = training()
    quality = testing(trained_model)
    print('Accuracy:', round(quality['correct']/quality['total'], 4))


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s'
    )
    main()
