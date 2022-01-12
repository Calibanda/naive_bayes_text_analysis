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
import numpy as np

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


# def get_dataset(
#         dataset_directory: pathlib.Path,
#         dataset_filename: str,
#         get_from_disk: bool = True
# ) -> pandas.DataFrame:
#
#     if get_from_disk:
#         try:
#             with open(dataset_filename, 'r') as f:
#                 dataset = pandas.read_csv(
#                     f,
#                     converters={'text': lambda x: x[2:-2].split("', '")}
#                 )
#                 dataset.rename(
#                     columns={'Unnamed: 0': 'index'},
#                     inplace=True
#                 )
#                 dataset.set_index('index', inplace=True)
#             return dataset
#         except FileNotFoundError:
#             pass
#
#     categories = [p.name for p in dataset_directory.iterdir()]
#     categories.sort()
#
#     dataset = pandas.DataFrame(columns=['label', 'text'])
#
#     for category in categories:
#         logging.info(f'Collecting {category}...')
#         category_path = dataset_directory / category
#         files_path = list(category_path.iterdir())
#         files_path.sort()
#         for message_path in files_path:
#             message = read_message(message_path)
#             word_list = clean_text(message)
#             if word_list:  # if word_list is not empty
#                 dataset = dataset.append(
#                     {'label': category, 'text': word_list},
#                     ignore_index=True
#                 )
#
#     # save sets for future use
#     logging.info(f'Save {dataset_filename} on disk.')
#     with open(dataset_filename, 'w') as f:
#         f.write(dataset.to_csv())
#
#     return dataset


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


# def get_word_count(
#         training_set: pandas.DataFrame,
#         get_from_disk: bool = True
# ) -> pandas.DataFrame:
#
#     if get_from_disk:
#         try:
#             with open('word_count_per_category.csv', 'r') as f:
#                 word_count_per_category = pandas.read_csv(f)
#             word_count_per_category.rename(
#                 columns={'Unnamed: 0': 'words'},
#                 inplace=True
#             )
#             word_count_per_category.set_index('words', inplace=True)
#             return word_count_per_category
#         except FileNotFoundError:
#             pass
#
#     word_count_per_category = {
#         category: {} for category in training_set['label'].unique()
#     }
#
#     for category, message in zip(training_set['label'], training_set['text']):
#         for word in message:
#             try:
#                 word_count_per_category[category][word] += 1
#             except KeyError:
#                 word_count_per_category[category][word] = 1
#
#     word_count_per_category = pandas.DataFrame(word_count_per_category)
#     word_count_per_category.fillna(1, inplace=True)  # replace NaN by 1
#
#     # save sets for future use
#     logging.info('Save word_count_per_category.csv on disk.')
#     with open('word_count_per_category.csv', 'w') as f:
#         f.write(word_count_per_category.to_csv())
#
#     # reload from disk for constancy
#     return get_word_count(training_set, get_from_disk=True)


# def get_probabilities(
#         word_count_per_category: pandas.DataFrame,
#         get_from_disk: bool = True
# ) -> pandas.DataFrame:
#     """
#     What data do I need (for each category):
#     P(category) = len(n_messages_category)/len(messages) (CONSIDERED AS
#         CONSTANT)
#     P(word | category) = p_word_given_category
#         = (n_word_given_category) / (sum_all_words_in_category)
#     """
#
#     if get_from_disk:
#         try:
#             with open('probabilities.csv', 'r') as f:
#                 probabilities = pandas.read_csv(f)
#                 probabilities.set_index('words', inplace=True)
#             return probabilities
#         except FileNotFoundError:
#             pass
#
#     word_count_per_category['sum'] = word_count_per_category.sum(axis=1)
#
#     probabilities = pandas.DataFrame(
#         None,
#         index=word_count_per_category.index,
#         columns=word_count_per_category.columns
#     )
#
#     for category in word_count_per_category.columns:
#         probabilities[category] \
#             = word_count_per_category[category] \
#             / word_count_per_category[category].sum()
#
#         logging.debug(  # verification
#             f'{category} sum verification: '
#             f'{probabilities[category].sum()}'
#         )
#
#     # save for future use
#     logging.debug('Save probabilities.csv on disk.')
#     with open('probabilities.csv', 'w') as f:
#         f.write(probabilities.to_csv())
#
#     return probabilities


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
        # model[category] = model[category] / len(category_word_list)

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

    # occurrence = {
    #     'occ': {word: message.count(word) for word in set(message)}
    # }
    # df_occ = pandas.DataFrame(occurrence)

    # logging.debug(df_occ)

    # prob = trained_model.pow(df_occ['occ'], axis=0)

    probabilities = sorted(  # sort by probability
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return probabilities
    # logging.debug(prob[:-1])

    # probabilities = {category: 1 for category in trained_model.columns}
    #
    # for word in message:
    #     pass

    # apparition = {}
    # classifier = trained_model[trained_model.index.isin(message)]
    #
    # logging.debug(f'{classifier.shape=}')
    # logging.debug(classifier.info())
    # logging.debug(classifier.head(10))
    #
    # classifier = classifier.prod()
    #
    # logging.debug(f'{classifier.shape=}')
    # logging.debug(classifier.head())

    # for word in message:
    #     if word in trained_model.index:


# def classify(
#         testing_set: pandas.DataFrame,
#         trained_model: pandas.DataFrame
# ) -> pandas.DataFrame:
#
#     # for category in trained_model.columns:
#     #     if category == 'sum':
#     #         continue
#     #     logging.debug(category)
#
#     #     testing_set[category] = trained_model.apply(
#     #         func=lambda row: math.prod([row[category]]) if row == ''
#     #     )
#
#     # result = classify_message(testing_set.loc[15, 'text'], trained_model)
#     # logging.debug(result)
#
#     testing_set = testing_set.apply(  # Apply "classify_message" on each rows
#         func=lambda row: classify_message(row['text'], trained_model),
#         axis=1,
#         result_type='expand',
#     )
#
#     logging.debug(f'{testing_set.shape=}')
#     logging.debug(testing_set)
#
#     # classify_message(testing_set.loc[15, 'text'], trained_model)
#     # logging.debug(f'{testing_set.shape=}')
#     # logging.debug(testing_set.head())
#
#     return testing_set


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
    #     category_counter = collections.Counter(category_word_list)
    #     model = pandas.concat(
    #         [model, pandas.Series(category_counter).rename(category)],
    #         axis=1  # Add as column
    #     )
    #     model[category] = model[category] / len(category_word_list)
    #
    # testing_set = get_dataset(TESTING_DIRECTORY, 'testing_set.csv')
    #
    # logging.debug(f'{testing_set.shape=}')
    # logging.debug(testing_set)
    #
    # classify(testing_set, trained_model)


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
