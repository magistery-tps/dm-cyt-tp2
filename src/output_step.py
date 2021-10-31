import logging
from transform_step import filter_words_lt

def plot_frequency(df, column):
    new_df = df[column].str.split(expand=True).stack().value_counts().reset_index()
    new_df.columns = ['Word', 'Frequency'] 
    new_df['Frequency'].plot(x='Words', y='Frequnecy', title='Words frequency', figsize=(6, 4))
    return df

def log_unique_words(df):
    logging.info("Unique - Source: {}, Response: {}".format(
        df['source'].unique().shape[0], 
        df['response'].unique().shape[0]
    ))
    return df

def log_source_into_response_and_vise_versa(df, unique):
    logging.info("Unique: {}, Source into response: {}, Response into source: {}".format(
        unique,
        sum(df['response'].isin(df['source'].unique())==unique), 
        sum(df['source'].isin(df['response'].unique())==unique)
    ))
    return df