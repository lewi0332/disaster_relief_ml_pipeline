import json
import plotly
import re
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
keywords_df = pd.read_csv('./data/keywords.csv', index_col=0)

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    agg_df = df.loc[(df['related'] == 1) & (df['request'] == 1)].groupby('genre').sum().drop(
        ['id', 'related', 'request'], axis=1).T.sort_values('direct', ascending=False)

    freq_data = []
    freq_menu = []
    for idex, elem in enumerate(keywords_df.columns):
        temp_x = list(keywords_df.loc[keywords_df[elem] > 0][elem].sort_values(ascending=False).index)
        temp_y = list(keywords_df.loc[keywords_df[elem] > 0][elem].sort_values(ascending=False))
        if idex == 0:
            freq_data.append(Bar(x=temp_x, y=temp_y, visible=True))
        else:
            freq_data.append(Bar(x=temp_x, y=temp_y, visible=False))
        visible = [False] * len(keywords_df.columns)
        visible[idex] = True
        freq_menu.append({'args': [{'visible': visible}], 'label': elem, 'method': 'restyle'})
    print(len(freq_menu))

    # create visuals
    graphs = [{
        'data': freq_data,
        'layout': {
            'title': 'Frequency of Words in Target Messages',
            'yaxis': {
                'title': "Frequency",
                'tickformat': '%'
            },
            'xaxis': {
                'title': "Type of Request"
            },
            'updatemenus': [{
                'buttons': freq_menu,
                'direction': 'down',
                'pad': {
                    'r': 10,
                    't': 10
                },
                'showactive': True
            }]
        }
    }, {
        'data': [
            Bar(x=agg_df.index, y=agg_df['direct'], name='Direct Messages'),
            Bar(x=agg_df.index, y=agg_df['news'], name='News Sources'),
            Bar(x=agg_df.index, y=agg_df['social'], name='Social Posts')
        ],
        'layout': {
            'title': 'Volume of Request Messages by Source',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Type of Request"
            },
        }
    }]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template('go.html', query=query, classification_result=classification_results)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()