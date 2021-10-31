from step_decorator import step_logger

@step_logger
def select_columns(df): return df[['cue','response']]

@step_logger
def rename_columns(df): return df.rename(columns={'cue': 'source'})

def filter_column_words_lt(df, column, lt=2): 
    return df.drop(df[df[column].str.len().lt(lt)].index)

@step_logger
def filter_words_lt(df, size_less_than):
    df = filter_column_words_lt(df,  'source', size_less_than)
    return filter_column_words_lt(df,  'response', size_less_than)

@step_logger
def filter_bidiredtional_associations(df):
    df   = df.drop( df[ df['response'].isin(df['source'].unique()  ) == False ].index)
    return df.drop( df[ df['source'  ].isin(df['response'].unique()) == False ].index)

@step_logger
def dropna(df): return df.dropna()