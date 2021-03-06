import numpy as np
import pandas as pd
import logging
from embedding import glove_word_embeddings
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from step import step_logger

def all_columns_contains(df, words):
    return df[df['source'].isin(words) & df['response'].isin(words)]

@step_logger
def select_columns(df): return df[['cue','R1']]

@step_logger
def n_top(df, top): return df[:top]

@step_logger
def rename_columns(df, columns):
    return df.rename(columns=columns)

def exclude_column_words_lt(df, column, lt=2): 
    return df.drop(df[df[column].str.len().lt(lt)].index)

@step_logger
def exclude_words_lt(df, size_less_than):
    df = exclude_column_words_lt(df,  'source', size_less_than)
    return exclude_column_words_lt(df,  'response', size_less_than)

@step_logger
def exclude_loops(df): return df[df.source != df.response]

@step_logger
def exclude_bidiredtional_associations(df):
    df   = df.drop( df[ df['response'].isin(df['source'].unique()  ) == False ].index)
    return df.drop( df[ df['source'  ].isin(df['response'].unique()) == False ].index)

@step_logger
def dropna(df): return df.dropna()

@step_logger
def exclude_stopwords(df, column, languages=stopwords.fileids()):
    if len(languages) == 1:
        stop_words = stopwords.words(languages[0])
        return df.drop(df[df[column].isin(stop_words)].index)        

    for language in languages:
        df = exclude_stopwords(df, column, languages=[language])

    return df

@step_logger
def to_unique_works(df):
    words = df['source'].values + df['response'].values
    rows = {word: True for word in words}
    return pd.DataFrame(data = rows.keys(), columns = ['word'])

@step_logger
def to_glove_word_embeddings(df, file_path):
    words = df['word'].values
    return {work: vector for (work, vector) in glove_word_embeddings(words, file_path)}

@step_logger
def lower(df):
    df['source']   = df['source'].str.lower()
    df['response'] = df['response'].str.lower()
    return df

@step_logger
def strip(df):
    return df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


@step_logger
def append_calculated_column(df, apply_fn, column_name):
    df[column_name] = df.apply(apply_fn, axis = 1)
    return df

@step_logger
def append_source_response_freqs(df):
    source_freq = df \
        .groupby(by=['source']) \
        .size() \
        .reset_index(name='source_freq')

    source_response_freq = df \
        .groupby(by=['source', 'response']) \
        .size() \
        .reset_index(name='source_response_freq')

    return df \
        .merge(source_freq, on='source') \
        .merge(source_response_freq, on=['source', 'response'])

@step_logger
def select(df, columns): return df[columns]

@step_logger
def distinct(df): return df.drop_duplicates()