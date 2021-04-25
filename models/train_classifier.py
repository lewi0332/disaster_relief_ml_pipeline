import sys
import pickle as pkl
# import libraries
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC


def load_data(database_filepath):
    """
    Loads data from database

    Input - database_filepath: filepath to sqlite database
    Output - X, y: Pandas DataFrames with Data and labels for training.
    """
    # get table name from filepath
    table = database_filepath.split('/')[1].split('.')[0]

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table, engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, y

def tokenize(text):
    """
    Tokenize with NLTK and removes URLs

    Input - text - Single string object with english message
    Output - list of lowercase, lemmatized word tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC()))
        ])
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__estimator__C': [.5, .75, 1.0],
        # 'clf__estimator__min_samples_split': [2, 3]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=3)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    results_df=pd.DataFrame(columns=['precision', 'recall', 'f1-score', 'support', 'accuracy'])
    for index, column in enumerate(y_test.columns):
        cr_dict = classification_report(y_test[column], y_pred[:,index], output_dict=True, labels=np.unique(y_pred[:,index]))
        cr_dict['weighted avg']['accuracy'] = accuracy_score(y_test[column], y_pred[:,index])
        results_df = results_df.append(pd.DataFrame(index=[column], data=cr_dict['weighted avg']))
    return results_df

def save_model(model, model_filepath):
    pkl.dump(model, open(model_filepath,'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        results_df = evaluate_model(model, X_test, y_test)
        print("Model results: ", results_df.mean())

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()