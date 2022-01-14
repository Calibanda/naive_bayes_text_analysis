#!/usr/bin/env python3
"""Naive Bayes classifier on the 20 Newsgroups data set.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright 2022, Clément SEIJIDO
Released under GNU General Public License v3.0 (GNU GPLv3)
e-mail clement@seijido.fr
"""

import logging
import re
import pathlib
import collections

import numpy as np
import pandas
import nltk

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


def read_message(message_path: str | pathlib.Path, encoding='utf-8') -> str:
    """Read a 20 Newsgroups message and remove headers and useless lines.

    Args:
        message_path (str | pathlib.Path): path of the file.
        encoding (str): encoding of the file. Default 'utf-8'.

    Returns:
        str: the 20 Newsgroups message.
    """

    wrong_lines_start = (
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
    )

    wrong_lines_end = (
        'writes:',
        'wrote:',
    )

    message = ''
    try:
        with open(message_path, 'r', encoding=encoding) as f:
            while f.readline() != '\n':  # skip headers
                pass

            for line in f:
                line = line.strip()
                if not line.startswith(wrong_lines_start) \
                        and not line.endswith(wrong_lines_end):
                    message += line + '\n'

        return message

    except UnicodeDecodeError:
        return read_message(message_path, encoding='iso-8859-1')


def tokenize_text(text: str) -> list[str]:
    """Tokenize and clean a given text: remove stop words, non alpha words, and
    words with length lower than 2. All words are also lemmatized
    (reduced to their lemma).

    Args:
        text (str): text to clean and tokenize.

    Returns:
        list[str]: cleaned words of the text.
    """

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
    """Collect 20 Newsgroups messages and create a model of all words
    apparition probabilities given Newsgroups categories.

    Returns:
        pandas.DataFrame: naive Bayes probability model.
    """

    try:  # If the model is on the disk, load it
        with open('trained_model.csv', 'r') as f:
            logging.info('Collect model from disk.')
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

    for index, category in enumerate(categories):
        logging.info(f'Collecting {category} ({index+1}/{len(category)})...')
        category_path = TRAINING_DIRECTORY / category
        files_path = list(category_path.iterdir())
        files_path.sort()

        category_word_list = []
        for message_path in files_path:
            message = read_message(message_path)
            category_word_list += tokenize_text(message)

        category_counter = collections.Counter(category_word_list)
        model = pandas.concat(
            [model, pandas.Series(category_counter).rename(category)],
            axis=1  # Add as column
        )
        n_words[category] = len(category_word_list)

    model.sort_index(inplace=True)
    model.index.name = 'words'

    # apply Laplace smoothing, compute probabilities and get log10
    model.fillna(0, inplace=True)  # replace NaN by 0
    model += 1
    for category in categories:
        model[category] = model[category] / n_words[category]

    model = np.log10(model)

    logging.debug(f'{model.shape=}')
    logging.debug(model)

    # save model for future use
    logging.info('Save model on disk.')
    with open('trained_model.csv', 'w') as f:
        f.write(model.to_csv())

    return model


def classify_message(
        message: str | list[str],
        model: pandas.DataFrame
) -> list[tuple[str, int]]:
    """Classify a message with the Bayes formula given a proper trained model.

    Args:
        message (str | list[str]): a message or a list of tokenized words.
        model (pandas.DataFrame): the trained model with all words
            apparition probabilities.

    Returns:
        list[tuple[str, int]]: the classification probabilities sorted.
    """

    if isinstance(message, str):
        message = tokenize_text(message)

    occurrences = collections.Counter(message)

    probabilities = {category: 0 for category in model.columns}

    for category in model.columns:
        for word, count in occurrences.items():
            try:
                probabilities[category] += model[category].loc[word] * count
            except KeyError:  # word is not in training model
                pass

    probabilities = sorted(  # sort by probability
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return probabilities


def testing(trained_model: pandas.DataFrame) -> dict[str, int]:
    """Test a naive Bayes model on 20 Newsgroups messages

    Args:
        trained_model (pandas.DataFrame): the trained model with all words
            apparition probabilities.
    Returns:
        dict[str, int]: the quality of the model: {'correct': X, 'total': X}
    """

    categories = [p.name for p in TESTING_DIRECTORY.iterdir()]
    categories.sort()

    quality = {'correct': 0, 'total': 0}

    for index, category in enumerate(categories):
        logging.info(f'Testing on {category} ({index+1}/{len(category)})...')
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
    logging.info(
        'Accuracy: ' + str(round(quality['correct']/quality['total'], 4))
    )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )
    main()
