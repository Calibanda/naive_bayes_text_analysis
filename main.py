#!/usr/bin/env python3
"""DESCRIPTION OF THE PROGRAM

(C) 2021 Clément SEIJIDO
Released under GNU General Public License v3.0 (GNU GPLv3)
e-mail clement@seijido.fr
"""
import logging
import re
import pathlib

import nltk
import pandas

pandas.set_option('display.max_columns', None)  # df.head() display all columns

TRAINING_DIRECTORY = pathlib.Path('20news-bydate/20news-bydate-train/')
TESTING_DIRECTORY = pathlib.Path('20news-bydate/20news-bydate-test/')


def get_dataset(
        dataset_directory: pathlib.Path,
        dataset_filename: str,
        get_from_disk: bool = True
) -> pandas.DataFrame:

    if get_from_disk:
        try:
            with open(dataset_filename, 'r') as f:
                dataset = pandas.read_csv(
                    f,
                    converters={'text': lambda x: x[2:-2].split("', '")}
                )
                dataset.rename(
                    columns={'Unnamed: 0': 'index'},
                    inplace=True
                )
                dataset.set_index('index', inplace=True)
            return dataset
        except FileNotFoundError:
            pass

    categories = [p.name for p in dataset_directory.iterdir()]
    categories.sort()

    dataset = pandas.DataFrame(columns=['label', 'text'])

    for category in categories:
        logging.info(f'Collecting {category}...')
        category_path = dataset_directory / category
        files_path = list(category_path.iterdir())
        files_path.sort()
        for message_path in files_path:
            message = read_message(message_path)
            word_list = clean_text(message)
            if word_list:  # if word_list is not empty
                dataset = dataset.append(
                    {'label': category, 'text': word_list},
                    ignore_index=True
                )

    # save sets for future use
    logging.info(f'Save {dataset_filename} on disk.')
    with open(dataset_filename, 'w') as f:
        f.write(dataset.to_csv())

    return dataset


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


def get_word_count(
        training_set: pandas.DataFrame,
        get_from_disk: bool = True
) -> pandas.DataFrame:

    if get_from_disk:
        try:
            with open('word_count_per_category.csv', 'r') as f:
                word_count_per_category = pandas.read_csv(f)
            word_count_per_category.rename(
                columns={'Unnamed: 0': 'words'},
                inplace=True
            )
            word_count_per_category.set_index('words', inplace=True)
            return word_count_per_category
        except FileNotFoundError:
            pass

    word_count_per_category = {
        category: {} for category in training_set['label'].unique()
    }

    for category, message in zip(training_set['label'], training_set['text']):
        for word in message:
            try:
                word_count_per_category[category][word] += 1
            except KeyError:
                word_count_per_category[category][word] = 1

    word_count_per_category = pandas.DataFrame(word_count_per_category)
    word_count_per_category.fillna(1, inplace=True)  # replace NaN by 1

    # save sets for future use
    logging.info('Save word_count_per_category.csv on disk.')
    with open('word_count_per_category.csv', 'w') as f:
        f.write(word_count_per_category.to_csv())

    # reload from disk for constancy
    return get_word_count(training_set, get_from_disk=True)


def get_probabilities(
        word_count_per_category: pandas.DataFrame,
        get_from_disk: bool = True
) -> pandas.DataFrame:
    """
    What data do I need (for each category):
    P(category) = len(n_messages_category)/len(messages) (CONSIDERED AS
        CONSTANT)
    P(word | category) = p_word_given_category
        = (n_word_given_category) / (sum_all_words_in_category)
    """

    if get_from_disk:
        try:
            with open('probabilities.csv', 'r') as f:
                probabilities = pandas.read_csv(f)
            return probabilities
        except FileNotFoundError:
            pass

    word_count_per_category['sum'] = word_count_per_category.sum(axis=1)

    for category in word_count_per_category.columns:
        name = category + '_proba'
        word_count_per_category[name] \
            = word_count_per_category[category] \
            / word_count_per_category[category].sum()

        logging.debug(  # verification
            f'{name} sum verification: '
            f'{word_count_per_category[name].sum()}'
        )

    probabilities = word_count_per_category.loc[
        :,  # all columns....
        word_count_per_category.columns.str.endswith('_proba')  # that match
    ]

    # save for future use
    logging.debug('Save probabilities.csv on disk.')
    with open('probabilities.csv', 'w') as f:
        f.write(probabilities.to_csv())

    return probabilities


def training() -> pandas.DataFrame:
    """For all categories
         For all files
           remove stop words
           lemmatisation
           vectorisation
           probabilities
    """

    training_set = get_dataset(TRAINING_DIRECTORY, 'training_set.csv')

    logging.debug(f'{training_set.shape=}')
    logging.debug(training_set.head())

    word_count_per_category = get_word_count(training_set)

    logging.debug(f'{word_count_per_category.shape=}')
    logging.debug(word_count_per_category.head())

    probabilities = get_probabilities(word_count_per_category)

    logging.debug(f'{probabilities.shape=}')
    logging.debug(probabilities.head())

    return probabilities


def testing(trained_model: pandas.DataFrame) -> None:
    categories = [p.name for p in TESTING_DIRECTORY.iterdir()]
    categories.sort()

    testing_set = get_dataset(TESTING_DIRECTORY, 'testing_set.csv')

    logging.debug(f'{testing_set.shape=}')
    logging.debug(testing_set.head())


def main():
    trained_model = training()
    testing(trained_model)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s'
    )
    main()
