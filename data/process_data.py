import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Reads two CSV files for disaster relief classification

    Input - 
        messages_filepath: filepath to csv with raw text from messages and tweets
        categories_filepath: filepath to csv file with category labels for 
                             previous message file
    Output - Single pandas DataFrame with combined data from files
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Prepares data for machine learning task

    Input: Pandas DataFrame with disaster response messages and 
           labeled classifications

    Output: Pandas DataFrame with expanded one-hot encoded label 
            classifications as integers
    """

    # Separate Category column
    categories = df['categories'].str.split(';', expand = True)
    # select the first row of the categories dataframe to get label names
    row = categories.values[0]
    category_colnames = list(map(lambda x: x[:-2], row))
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Drop any columns that zero positive results.
    zero_cols = [ col for col, is_zero in ((categories == 0).sum() == categories.shape[0]).items() if is_zero ]
    categories.drop(zero_cols, axis=1, inplace=True)

    # Drop original catgories column and replace with new expanded and cleaned columns
    df.drop('categories', axis = 1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Drop Duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Saves cleaned data to sqlite database. 

    Input - Cleaned pandas DataFrame ready for ML task
    Output - Message of outcome of data loaded to database
    """
    table = database_filename.split('/')[1].split('.')[0]
    try:
        engine = create_engine(f'sqlite:///{database_filename}')
        df.to_sql(table, engine, index=False, if_exists='replace')
        return f"Successfully added {len(df)} rows to {database_filename}"
    except Exception as e: 
        return print(e)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()