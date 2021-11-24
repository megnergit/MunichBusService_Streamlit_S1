import pandas as pd
import numpy as np
import urllib
from pathlib import Path
import config
import requests
from geopandas.tools import geocode
import re
from monkeylearn import MonkeyLearn
import plotly.graph_objs as go
from plotly.colors import sample_colorscale
from wordcloud import WordCloud
from folium import Circle
from datetime import datetime
import config
import ast
import leafmap.foliumap as folium
import time
import pdb
import pretty_errors
# ====================================
# Authentication
# ====================================
AK = config.API_KEY  # not really necessary
AKS = config.API_KEY_SECRET  # not really necessary
BT = config.BEARER_TOKEN

MKL_AK = config.MONKEYLEARN_API_KEY
MKL_ST_MODEL_ID = config.MONKEYLEARN_SENTIMENT_MODEL_ID
MKL_EX_MODEL_ID = config.MONKEYLEARN_KEYWORD_EXTRACTOR_MODEL_ID

# ====================================
# house keeping
# ====================================


def show_all() -> None:
    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 99
    pd.options.display.expand_frame_repr = True
#    pd.options.display.large_repr = 'info'
    pd.options.display.large_repr = 'truncate'


def get_lat_lon(landmark='Marienplatz'):
    # Marienplatz
    # (11.57540 48.13714)
    # centroid Nunich
    #  11.525549293315658, 48.1548901
    results_geocode = geocode(landmark)
    lon = results_geocode['geometry'].x.values[0]
    lat = results_geocode['geometry'].y.values[0]
    return lat, lon


def color_func(sentiment):
    if sentiment == 'Negative':
        return 'maroon'
    elif sentiment == 'Neutral':
        return 'gold'
    else:
        return 'turquoise'

# ====================================
# Collecting twitters
# ====================================


def query_text_mbs(BT):

    QUERY = 'q=Bus'
    QUERY2 = 'q=mvv'
    RECENT = 'result_type=recent'
    LANG = 'lang=de'
    # LANG = 'lang=en'
    # EXTENDED = 'tweet_mode=extended'
    COUNT = 'count=100'
    LAND_MARK = 'Marienplatz'
    RADIUS = 20.0  # km
    RT = urllib.parse.quote_plus('-is:retweet')
    URL_TWITTER_API = "https://api.twitter.com/1.1/search/tweets.json?"
    HEADERS = {"Authorization": "Bearer {}".format(BT)}
    lat, lon = get_lat_lon(landmark=LAND_MARK)
    GEO = 'geocode='+str(lat)+','+str(lon)+','+str(RADIUS)+'km'

#    URL = URL_TWITTER_API + QUERY+'&'+GEO+'&'+LANG+'&'+COUNT+'&'+RECENT
#    URL = URL_TWITTER_API + GEO+'&'+LANG+'&'+COUNT+'&'+RT
#    URL = URL_TWITTER_API + GEO+'&'+LANG+'&'+COUNT
    URL = URL_TWITTER_API + QUERY+'&'+GEO+'&'+LANG+'&'+COUNT+'&'+RT+'&'+RECENT
    return URL, HEADERS

# --------------------------------------------
# def collect_tweet(BT):
#     URL, HEADERS = query_text_mbs(BT)
#     response = requests.request("GET", URL, headers=HEADERS).json()
#     return response

# --------------------------------------------
# polling function : curate tweets collections
# --------------------------------------------


def pre_polling(outfile,
                LOG_FILE='./log/log_file.txt',
                LOG_FILE_COLOR='./log/log_file_color.txt'):

    # -----------------------------
    # if there is outfile, read it
    # to get length of records
    # -----------------------------
    max_id_str = '0'
    initial_length = 0

    # -------------------------
    if Path(outfile).exists():
        df_prev = pd.read_csv(outfile, dtype={'id': np.int64},
                              parse_dates=['created_at'])

        backup_file = Path(str(outfile).replace('.csv', '') + '_' +
                           datetime.now().strftime('%Y-%m-%d') + '.csv')
        df_prev.to_csv(backup_file, index=False)

        initial_length = len(df_prev)
        max_id = df_prev['id'].max()
        max_id_str = str(max_id)

        del df_prev

    # -------------------------
    # log
    m = max_id_str
    print(f'\033[93mcurrent record length\033[0m {initial_length} \
{m[0:4]} {m[4:8]} {m[8:12]} {m[12:16]} {m[16:20]}')

    with open(LOG_FILE, 'a') as log_file:
        print(
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} \
current record length {initial_length} {max_id_str}',
            file=log_file)

    with open(LOG_FILE_COLOR, 'a') as log_file_color:
        print(
            f'<b style="color:salmon">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} </b> \
<b style="color:plum"> current record length</b> \
{initial_length} {m[0:4]} {m[4:8]} {m[8:12]} {m[12:16]} {m[16:20]}',
            file=log_file_color)

# # steelblue skyblue plum orchid darkorchid pink lightpink
#     else:
#         initial_length = 0
# #        since_id_str = '0'
# #        since_id = 0
# #        max_id = 0
#         max_id_str = '0'

    return max_id_str, initial_length

# --------------------------------------------


def polling_tweets(BT, DEEPL_AK,
                   outfile='./data/tweet_bus_de_en.csv',
                   n_stopper=3, t_sleep=1,
                   MKL_AK='',
                   MKL_ST_MODEL_ID='',
                   MKL_EX_MODEL_ID='',
                   DATA_DIR='./data',
                   LOG_FILE='./log/log_file.txt',
                   LOG_FILE_COLOR='./log/log_file_color.txt'
                   ) -> None:

    max_id_str, initial_length = pre_polling(outfile,
                                             LOG_FILE=LOG_FILE,
                                             LOG_FILE_COLOR=LOG_FILE_COLOR)

    revised_length = initial_length
    df_list = []
    URL, HEADERS = query_text_mbs(BT)

    for i in range(n_stopper):
        prev_length = revised_length
        new_max_str, revised_length, df_list = fetch_tweets(outfile,
                                                            URL, HEADERS, max_id_str,
                                                            df_list,
                                                            revised_length,
                                                            DEEPL_AK,
                                                            LOG_FILE,
                                                            LOG_FILE_COLOR)

        # print(' ')
        # print('len(df_list)', len(df_list))
        # print('revised_length ', revised_length)
        # print('old_max_str ', max_id_str)
        # print('new_max_str ', new_max_str)
        # pdb.set_trace()
        # store relevant files
#        if int(new_max_str) > int(max_id_str):
        if revised_length > prev_length:
            preprocess_mbs(outfile,
                           MKL_AK, MKL_ST_MODEL_ID, MKL_EX_MODEL_ID,
                           DATA_DIR)

        max_id_str = new_max_str

        if i == (n_stopper - 1):
            # ----
            # logging purpose only
            post_polling(new_max_str, revised_length, LOG_FILE=LOG_FILE,
                         LOG_FILE_COLOR=LOG_FILE_COLOR)

            return  # exit without waiting

        time.sleep(t_sleep)

    # post_polling(new_max_str, revised_length, LOG_FILE=LOG_FILE,
    #              LOG_FILE_COLOR=LOG_FILE_COLOR)

#    return

# --------------------------------------------


def fetch_tweets(outfile, URL, HEADERS, max_id_str,
                 df_list,
                 revised_length,
                 DEEPL_AK,
                 LOG_FILE,
                 LOG_FILE_COLOR):

    t1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_max_str = max_id_str

    r = requests.request("GET", URL+'&since_id='+max_id_str,
                         headers=HEADERS).json()
    if len(r['statuses']) == 0:
        new_max_str = r['search_metadata']['max_id_str']
        # print('')
        # print('zero r')
        # print('old_max_str ', max_id_str)
        # print('new_max_str ', new_max_str)

        return new_max_str, revised_length, df_list

    clean_r = clean_response(r)
    if len(clean_r['statuses']) == 0:
        new_max_str = clean_r['search_metadata']['max_id_str']
        # print('')
        # print('zero clean_r')
        # print('old_max_str ', max_id_str)
        # print('new_max_str ', new_max_str)

        return new_max_str, revised_length, df_list

    # revise new_max_str
    new_max_str = clean_r['search_metadata']['max_id_str']
    # ---------------------------
    # store file
    # print('inside fetch', len(clean_r['statuses']))
    # print('old_max_str', max_id_str)
    # print('new_max_str', new_max_str)
    df = response_to_csv(clean_r)
    # =============================================
    # DEEPL
    # --------------------------------------------
    df, usage = add_text_en(df, DEEPL_AK)
    df_list.append(df)

    df = pd.concat(df_list, axis=0)
    df.drop_duplicates(subset=['id'], inplace=True)
#    length_after = len(df)

    df.sort_values(['id'], inplace=True)
#    print(df.info())
#    pdb.set_trace()
    column_order = ['id', 'created_at', 'created_at_tz', 'geo', 'place', 'coordinates',
                    'text', 'text_en', 'truncated', 'name', 'screen_name', ]
    if Path(outfile).exists():
        df.to_csv(outfile, mode='a',
                  columns=column_order,
                  header=False, index=False)
    else:
        df.to_csv(outfile,  columns=column_order, index=False)

#    print('')
#    print('in fetch')
#    print(df.tail())
#    pdb.set_trace()

    retrieved_length = len(df)
    revised_length = revised_length + retrieved_length
    # ---------------------------
    # log
    m = new_max_str

    print(f'\033[33m{t1} \033[91mmax_id \
\033[0m{m[0:4]} {m[4:8]} {m[8:12]} {m[12:16]} {m[16:20]} \
\033[96mret: \033[0m{retrieved_length:3} \
\033[94mtot: \033[0m{revised_length: 5}\033[0m')

    with open(LOG_FILE, 'a') as log_file:
        print(
            f'{t1} max_id {m[0:4]} {m[4:8]} {m[8:12]} {m[12:16]} {m[16:20]} \
ret: {retrieved_length:3} tot: {revised_length:5}', end='', file=log_file)

    with open(LOG_FILE_COLOR, 'a') as log_file_color:
        print(
            f'<b style="color:salmon">{t1} </b> \
                    <b style="color:indianred">max_id</b> \
{m[0:4]} {m[4:8]} {m[8:12]} {m[12:16]} {m[16:20]} \
<b style="color:deeppink">ret</b>: {retrieved_length: 3} \
<b style="color:firebrick">tot</b>: {revised_length: 5}',
            file=log_file_color)

    return new_max_str, revised_length, df_list

 # --------------------------------------------


def post_polling(max_id_str, revised_length, LOG_FILE, LOG_FILE_COLOR):
    # loggin only

    # terminal / text log file / for web page
    m = max_id_str

    print(f'\033[94mrevised record length\033[0m {revised_length} \
{m[0:4]} {m[4:8]} {m[8:12]} {m[12:16]} {m[16:20]}')

    with open(LOG_FILE, 'a') as log_file:
        print(
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} \
revised record length {revised_length} {max_id_str}',
            file=log_file)

    with open(LOG_FILE_COLOR, 'a') as log_file_color:
        print(
            f'<b style="color:salmon">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} </b> \
<b style="color:slateblue">revised record length </b>\
{revised_length} {m[0:4]} {m[4:8]} {m[8:12]} {m[12:16]} {m[16:20]}',
            file=log_file_color)

 # --------------------------------------------


def preprocess_mbs(outfile,
                   MKL_AK, MKL_ST_MODEL_ID, MKL_EX_MODEL_ID,
                   DATA_DIR):

    input_file = outfile
    df = pd.read_csv(input_file,
                     parse_dates=['created_at'])

#    pdb.set_trace()  # already broken
    # =============================================
    # monkey learn
    # --------------------------------------------
    df_stx = get_mkl_st_dummy(df, MKL_AK, MKL_ST_MODEL_ID)
    df_kex = get_mkl_ex_dummy(df_stx, MKL_AK, MKL_EX_MODEL_ID)
    df_kex.to_csv(DATA_DIR/'mbs_kex.csv', index=False)

#    print('df_kex inside preprocess_mbs')
#     print(df_kex.info())
#     pdb.set_trace()
    # =============================================
    # aggregate
    # --------------------------------------------
    df_pn = add_sentiment_digit(df_kex)
    df_pn.to_csv(DATA_DIR/'mbs_pn.csv', index=False)

#    print('df_pn')
#    print(df_pn.info())
#    pdb.set_trace()

#    df_pn['created_at'] = df_pn['created_at'].astype('datetime64[ns]')
    df_pn['created_at_tz'] = df_pn['created_at_tz'].astype('datetime64[ns]')
    #    df_agg = aggregate_sentiment(df_pn, freq='12H')

    df_agg = aggregate_sentiment_tz(df_pn, freq='12H')
 #   print('df_agg')
#    print(df_agg.info())
#    pdb.set_trace()

    # need to store index as well, since that is the datetime
    df_agg.to_csv(DATA_DIR/'mbs_agg.csv')

    # =============================================
    # geo
    # --------------------------------------------

    df_geo = extract_place(df_kex)
    df_geo.to_csv(DATA_DIR/'mbs_geo.csv', index=False)

# --------------------------------------------
# steelblue skyblue plum orchid darkorchid
# --------------------------------------------


def clean_response(response):

    EXCLUDE = ['MVGticker', 'S-Bahn München']

    new_response = []
    for i in response['statuses']:
        if i['user']['name'] not in EXCLUDE:
            i['text'] = re.sub(
                r"(@[A-Za-z0–9_]+)|[^\w\s]|#|http\S+", "", i['text'])

            i['text'] = re.sub(r"&", " and ", i['text'])
            new_response.append(i)

    return dict(statuses=new_response,
                search_metadata=response['search_metadata'])


# --------------------------------------------

def response_to_csv(response: dict) -> pd.DataFrame:

    # keys_all = response['statuses'][0].keys()
    # print(keys_all)

    keys = ['id', 'created_at',  'geo', 'place',
            'coordinates', 'text', 'truncated']
    user_keys = ['name', 'screen_name']

    df = pd.DataFrame()
    for k in keys:
        df[k] = [r[k] for r in response['statuses']]

    for u_k in user_keys:
        df[u_k] = [r['user'][u_k] for r in response['statuses']]

    df['created_at'] = df['created_at'].astype('datetime64[ns]')
    df['created_at_tz'] = [t.tz_localize('UTC').tz_convert(
        'Europe/Berlin') for t in df['created_at']]

#    print(df.info())
#    pdb.set_trace()
    return df


# --------------------------------------------

def extract_place(df):
    df_place = df.loc[~df['place'].isna(), :].copy()
    place_list = df_place['place'].to_list()

    if type(place_list[0]) == type('s'):
        place_list = [ast.literal_eval(d) for d in place_list]

    lon_list = []
    lat_list = []
    for p in place_list:
        p_array = np.array(p['bounding_box']['coordinates'][0])
        lon, lat = p_array.mean(axis=0)
        lon_list.append(lon)
        lat_list.append(lat)

    df_place['lon'] = lon_list
    df_place['lat'] = lat_list

    return df_place

# --------------------------------------------


def show_text(response) -> None:
    print(f'\033[33m=\033[0m'*40)
    for i in response['statuses']:
        if i['user']['name'] not in ['MVGticker', 'S-Bahn München']:
            i['text'] = re.sub(
                r"(@[A-Za-z0–9_]+)|[^\w\s]|#|http\S+", "", i['text'])


def show_response(response) -> None:

    for i in response['statuses']:
        for k, v in i.items():
            if k != 'user':
                print(f'\033[33m{k} \033[0m: {v}')
            else:
                for k2, v2 in v.items():
                    print(f'   \033[31m{k2} \033[0m: {v2}')
        print(f'\033[37m='*40)

    print('='*40)

# ====================================
# MonkeyLearn
# ====================================


def get_mkl_st(df, MKL_AK, MKL_ST_MODEL_ID):

    ml = MonkeyLearn(MKL_AK)
    df_stx = df.copy()
    ml_st_response = ml.classifiers.classify(
        model_id=MKL_ST_MODEL_ID,
        data=df_stx['text_en'].to_list()
    )
    stx = ml_st_response.body
    df_stx['sentiment'] = [s['classifications'][0]['tag_name'] for s in stx]
    df_stx['confidence'] = [s['classifications'][0]['confidence'] for s in stx]

    return df_stx


def sort_mkl_st(df):
    # expect :  df_stx
    # add columns ['sentiment_digit'], ['confidence_digit'],
    # remove neutral
    df_pn = df.copy()

    df_pn['sentiment_digit'] = df_pn['sentiment']
    df_pn['sentiment_digit'].replace(
        ['Positive', 'Neutral', 'Negative'], [1, 0, -1], inplace=True)

    df_pn['confidence_digit'] = df_pn['confidence']
    ix = df_pn['sentiment'] == 'Negative'
    df_pn.loc[ix, 'confidence_digit'] = -1.0 * df_pn.loc[ix, 'confidence']

    df_pn['confidence_digit_zero_neutral'] = df_pn['confidence'] * \
        df_pn['sentiment_digit']

    return df_pn.sort_values(['sentiment_digit', 'confidence_digit'])


def add_sentiment_digit(df):
    # expect :  df_stx
    # add columns ['sentiment_digit'], ['confidence_digit'],

    # remove neutral
    df_pn = df.copy()

    df_pn['sentiment_digit'] = df_pn['sentiment']
    df_pn['sentiment_digit'].replace(
        ['Positive', 'Neutral', 'Negative'], [1, 0, -1], inplace=True)

    df_pn['confidence_digit'] = df_pn['confidence']
    ix = df_pn['sentiment'] == 'Negative'
    df_pn.loc[ix, 'confidence_digit'] = -1.0 * df_pn.loc[ix, 'confidence']

    df_pn['confidence_digit_zero_neutral'] = df_pn['confidence'] * \
        df_pn['sentiment_digit']

    return df_pn
# ====================================
# keyword extractor
# ====================================


def get_mkl_ex(df, MKL_AK, MKL_EX_MODEL_ID):
    ml = MonkeyLearn(MKL_AK)
    df_kex = df.copy()

    ml_ex_response = ml.extractors.extract(
        model_id=MKL_EX_MODEL_ID,
        data=df['text_en'].to_list())

    ex = ml_ex_response.body
    df_kex['keyword'] = [
        e['extractions'][0]['parsed_value'] for e in ex]

    return df_kex

# ============================================
# ============================================
# visualizations
# ============================================


def visualize_pn(df, size=640, vertical=True):
    n_last = 30

    df_pn = df.loc[df['sentiment'] != 'Neutral', :].copy()
    df_pn = df_pn.sort_values('created_at').tail(n_last)
    df_pn = df_pn.sort_values(['sentiment_digit', 'confidence_digit'])

    cs = 'Agsunset'  # 'Earth', 'Geyser', 'Viridis', 'thermal', 'solar', 'balance'
    score = df_pn['confidence_digit'].mean()
    score_scale = 0.5 * (score + 1.0)
    score_color = sample_colorscale(cs, score_scale)[0]

    axis_dict1 = dict(tickfont=dict(size=16),
                      tickmode='array',
                      ticks='inside',
                      tickvals=list(range(len(df_pn))),
                      ticktext=df_pn['keyword'])

    axis_dict2 = dict(autorange=False,
                      title=dict(text='<- Negatve | Positive ->',
                                 font=dict(size=16)),
                      range=[-1.01, 1.01],
                      tickfont=dict(size=16))

    if vertical:
        trace = go.Bar(x=df_pn['confidence_digit'],
                       y=list(range(len(df_pn))),
                       orientation='h',
                       marker_color=df_pn['confidence_digit'],
                       marker=dict(colorscale='Agsunset'),
                       hovertext=df_pn['text_en'],
                       hoverinfo='text',
                       hoverlabel=dict(font=dict(size=20)))

        xdict = axis_dict2
        ydict = axis_dict1
        height = size * 1.2
        width = size

    else:
        trace = go.Bar(y=df_pn['confidence_digit'],
                       x=list(range(len(df_pn))),
                       orientation='v',
                       marker_color=df_pn['confidence_digit'],
                       marker=dict(colorscale='Agsunset'),
                       hovertext=df_pn['text_en'],
                       hoverinfo='text',
                       hoverlabel=dict(font=dict(size=20)))

        xdict = axis_dict1
        ydict = axis_dict2
        height = size
        width = size * 2

    layout = go.Layout(
        legend=dict(yanchor='bottom', xanchor='right'),
        legend_title='Sentiment',
        xaxis=xdict,
        yaxis=ydict,
        font=dict(size=24),
        height=size,  # width=width,
        margin=dict(l=0, r=0, t=0, b=0),)

    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    return fig

# ============================================


def visualize_agg(df, size):

    xmin = df.index[0] - df.index.freq
    xmax = df.index[-1] + df.index.freq

    cs = 'Agsunset'  # 'Earth', 'Geyser', 'Viridis', 'thermal', 'solor', 'balance'
    positive_color = sample_colorscale(cs, 0.9)[0]
    negative_color = sample_colorscale(cs, 0.1)[0]
    overall_color = sample_colorscale(cs, 0.5)[0]

    score = df.iloc[-1]['confidence_mean']
    score_scale = 0.5 * (score + 1.0)
    score_color = sample_colorscale(cs, score_scale)[0]

    trace_zero = go.Scatter(
        x=[xmin, xmax],
        y=[0, 0],
        mode='lines',
        line=dict(width=8, color='lightgray'),
        opacity=0.8

    )

    trace_overall = go.Scatter(
        x=df.index.to_series(),
        y=df['confidence_mean'],
        mode='lines+markers',
        marker=dict(size=20),
        line=dict(width=8, color=overall_color),
        opacity=0.8

    )

    trace_positive = go.Scatter(
        x=df.index.to_series(),
        y=df['positive_mean'],
        mode='lines+markers',
        marker=dict(size=14),
        line=dict(width=4, color=positive_color),
        opacity=0.3

    )

    trace_negative = go.Scatter(
        x=df.index.to_series(),
        y=df['negative_mean'],
        mode='lines+markers',
        marker=dict(size=14),
        line=dict(width=4, color=negative_color),
        opacity=0.3
    )

    xdict = dict(tickfont=dict(size=20),
                 autorange=False,
                 range=[xmin, xmax])

    ydict = dict(autorange=False,
                 range=[-1.01, 1.01],
                 tickfont=dict(size=20))

    layout = go.Layout(
        height=size * 0.75,
        showlegend=False,
        xaxis=xdict,
        yaxis=ydict,
        font=dict(size=24),
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[dict(font=dict(size=24, color=positive_color),
                          text='Positive',
                          showarrow=False,
                          x=df.index.to_series()[0], y=-0.2,
                          xanchor='left'),

                     dict(font=dict(size=28, color=overall_color),
                          text='Overvall',
                          showarrow=False,
                          x=df.index.to_series()[0], y=-0.35,
                          xanchor='left'),

                     dict(font=dict(size=24, color=negative_color),
                          text='Negative',
                          showarrow=False,
                          x=df.index.to_series()[0], y=-0.5,
                          xanchor='left'),

                     # latest score
                     dict(font=dict(size=24),
                          text='Sentiment Score',
                          showarrow=False,
                          x=df.index.to_series()[-1], y=0.9,
                          xanchor='right',
                          yanchor='top'),

                     dict(font=dict(size=40, color=score_color),
                          text=f'{score:5.2}',
                          showarrow=False,
                          x=df.index.to_series()[-1], y=0.8,
                          xanchor='right',
                          yanchor='top'),

                     ])

    data = [trace_zero, trace_negative, trace_positive, trace_overall]
    fig = go.Figure(data=data, layout=layout)
    return fig

# --------------------------------------------


def visualize_count(df, size):

    xmin = df.index[0] - df.index.freq
    xmax = df.index[-1] + df.index.freq
    ymax = max(df['count_positive'].max(), df['count_negative'].max())

    cs = 'Agsunset'  # 'Earth', 'Geyser', 'Viridis', 'thermal', 'solor', 'balance'
    positive_color = sample_colorscale(cs, 0.9)[0]
    negative_color = sample_colorscale(cs, 0.1)[0]
    overall_color = sample_colorscale(cs, 0.5)[0]

    trace_zero = go.Scatter(
        x=[xmin, xmax],
        y=[0, 0],
        mode='lines',
        line=dict(width=8, color='lightgray'),
        opacity=0.8

    )

    trace_positive = go.Bar(
        x=df.index.to_series(),
        y=df['count_positive'],
        text=df['count_positive'],
        marker=dict(color=positive_color),
        opacity=0.3,
        texttemplate='%{text:A}',
        textposition='inside'
    )

    trace_negative = go.Bar(
        x=df.index.to_series(),
        y=df['count_negative'],
        text=df['count_negative'],
        marker=dict(color=negative_color),
        opacity=0.3,
        texttemplate='%{text:A}',
        textposition='inside'
    )

    xdict = dict(tickfont=dict(size=20),
                 autorange=False,
                 range=[xmin, xmax])

    ydict = dict(
        range=[0, ymax * 1.2],
        tickfont=dict(size=20))

    layout = go.Layout(
        height=size * 0.5,
        showlegend=False,
        xaxis=xdict,
        yaxis=ydict,
        font=dict(size=24),
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[dict(font=dict(size=24, color=positive_color),
                          text='Positive',
                          showarrow=False,
                          x=df.index.to_series()[0], y=ymax*1.0,
                          xanchor='left'),

                     dict(font=dict(size=24, color=negative_color),
                          text='Negative',
                          showarrow=False,
                          x=df.index.to_series()[0], y=ymax*0.85,
                          xanchor='left'),
                     ])

    data = [trace_zero, trace_negative, trace_positive]
    fig = go.Figure(data=data, layout=layout)
    return fig


# ====================================


def aggregate_sentiment(df, freq='60S'):

    df_agg = pd.DataFrame()
    df_agg['confidence_mean'] = df.groupby(pd.Grouper(
        key='created_at', freq=freq)).mean()['confidence_digit']
    df_agg['confidence_std'] = df.groupby(pd.Grouper(
        key='created_at', freq=freq)).std()['confidence_digit']
    df_agg['count'] = df.groupby(pd.Grouper(
        key='created_at', freq=freq)).count()['confidence_digit']

    df_agg['positive_mean'] = df[df['sentiment'] == 'Positive'].groupby(
        pd.Grouper(key='created_at', freq=freq)).mean()['confidence_digit']

    df_agg['count_positive'] = df[df['sentiment'] == 'Positive'].groupby(
        pd.Grouper(key='created_at', freq=freq)).count()['confidence_digit']

    df_agg['negative_mean'] = df[df['sentiment'] == 'Negative'].groupby(
        pd.Grouper(key='created_at', freq=freq)).mean()['confidence_digit']

    df_agg['count_negative'] = df[df['sentiment'] == 'Negative'].groupby(
        pd.Grouper(key='created_at', freq=freq)).count()['confidence_digit']

    df_agg['count_neutral'] = df[df['sentiment'] == 'Neutral'].groupby(
        pd.Grouper(key='created_at', freq=freq)).count()['confidence_digit']

    return df_agg
# --------------------------------------------------


def aggregate_sentiment_tz(df, freq='60S'):

    df_agg = pd.DataFrame()
    df_agg['confidence_mean'] = df.groupby(pd.Grouper(
        key='created_at_tz', freq=freq)).mean()['confidence_digit']
    df_agg['confidence_std'] = df.groupby(pd.Grouper(
        key='created_at_tz', freq=freq)).std()['confidence_digit']
    df_agg['count'] = df.groupby(pd.Grouper(
        key='created_at_tz', freq=freq)).count()['confidence_digit']

    df_agg['positive_mean'] = df[df['sentiment'] == 'Positive'].groupby(
        pd.Grouper(key='created_at_tz', freq=freq)).mean()['confidence_digit']

    df_agg['count_positive'] = df[df['sentiment'] == 'Positive'].groupby(
        pd.Grouper(key='created_at_tz', freq=freq)).count()['confidence_digit']

    df_agg['negative_mean'] = df[df['sentiment'] == 'Negative'].groupby(
        pd.Grouper(key='created_at_tz', freq=freq)).mean()['confidence_digit']

    df_agg['count_negative'] = df[df['sentiment'] == 'Negative'].groupby(
        pd.Grouper(key='created_at_tz', freq=freq)).count()['confidence_digit']

    df_agg['count_neutral'] = df[df['sentiment'] == 'Neutral'].groupby(
        pd.Grouper(key='created_at_tz', freq=freq)).count()['confidence_digit']

    return df_agg

# ====================================


def add_text_en(df, DEEPL_AK):

    auth_key = 'auth_key='+DEEPL_AK
    source_lang = 'source_lang=DE'
    target_lang = 'target_lang=EN'

    URL_DEEPL = "https://api-free.deepl.com/v2/translate?"
    URL_DEEPL_USAGE = "https://api-free.deepl.com/v2/usage?"

    DEEPL_QUERY = '&'.join([auth_key, source_lang, target_lang])
    URL_DEEPL_QUERY = URL_DEEPL + DEEPL_QUERY

    df['text_en'] = df['text'].copy()
    df['text_en'] = ''

    i = 0
    n_text = 50
    while n_text*i < len(df):
        text = '&' + '&'.join(
            ['text='+d for d in df.loc[n_text*i:n_text*(i+1)-1, 'text']])
        response = requests.request("POST", URL_DEEPL_QUERY+text).json()
        df.loc[n_text*i:n_text*(i+1)-1, 'text_en'] = [t['text']
                                                      for t in response['translations']]
        i = i + 1

    # check how much credit still remains
    URL_DEEPL_USAGE_QUERY = URL_DEEPL_USAGE + auth_key
    usage = requests.request("POST", URL_DEEPL_USAGE_QUERY).json()
    print(
        f"\033[91mUsed:\033[93m {usage['character_count']}\033[0m/{usage['character_limit']}")

    return df, usage

# # ================================================
# dummy functions until MonkeyLaern available again
# --------------------------------------------------


def get_mkl_st_dummy(df, MKL_AK, MKL_ST_MODEL_ID):

    df_stx = df.copy()

    classification = np.random.choice(
        ['Positive', 'Negative', 'Neutral'], size=len(df))
    confidence = np.random.random(size=len(df))
    df_stx['sentiment'] = classification
    df_stx['confidence'] = confidence

    df_stx['sentiment_digit'] = df_stx['sentiment'].copy()
    df_stx['sentiment_digit'].replace(
        ['Positive', 'Neutral', 'Negative'], [1, 0, -1], inplace=True)

    return df_stx


# ====================================
def get_mkl_ex_dummy(df, MKL_AK, MKL_EX_MODEL_ID):
    df_kex = df.copy()

    df_kex['keyword'] = [[w.split()[np.array([len(s) for s in w.split()]).argmax()]
                          for w in df.loc[i:i, 'text_en']][0] for i in range(len(df))]

    return df_kex


# ============================================
# play with wordcloud
# --------------------------------------------


def create_wordcloud(df, size=640):
    width = size
    height = size  # quadratisch
    cm = 'inferno'  # 'plasma'  # 'Dark2' # 'Spectral'  # 'Plasma' 'RdBu', 'Pastel2'
    text = ' '.join(df['keyword'].to_list())

    wordcloud = WordCloud(background_color='white',
                          colormap=cm,  # from matplotlib
                          prefer_horizontal=0.9,
                          width=width, height=height,
                          mode='RGBA').generate(text)
    return wordcloud


def visualize_wc(wc):
    #    layout = go.Layout(width=wc.width, height=wc.height,
    layout = go.Layout(margin=dict(l=0, r=0, t=20, b=0),
                       xaxis=dict(visible=False),
                       yaxis=dict(visible=False))
    trace = go.Image(z=wc, xaxis='x', yaxis='y')
    fig = go.Figure(data=[trace], layout=layout)
    return fig

# --------------------------------------------


def plot_sentiment(df_stx):

    df_geo = extract_place(df_stx)
    center = get_lat_lon()
    tiles = 'openstreetmap'
    # tiles = 'Stamen Terrain'
    # tiles = 'cartodbpositron'
    # tiles = 'mapquestopen'
    zoom = 12
    m_1 = folium.Map(location=center, tiles=tiles, zoom_start=zoom)

    dump = [Circle([r['lat'], r['lon']],
                   radius=10 ** (2.0 + r['confidence']),
                   color='lightgray',
                   tooltip=r['text_en'],
                   fill_color=color_func(r['sentiment']),
                   fill_opacity=0.5,
                   fill=True).add_to(m_1)
            for i, r in df_geo.iterrows()]

    return m_1

# ====================================
# utility : quickly rewind data file
# ====================================
#  outfile = 'data/tweet_bus_de_en.csv'
# remove_duplicates(outfile)


def remove_duplicates(outfile):
    df = pd.read_csv(outfile, parse_dates=['created_at'])
    print(len(df))
    df.drop_duplicates(subset=['id'], inplace=True)
    print(len(df))

# convert utc to tz = 'Europe/Berlin'
    df['created_at_tz'] = [t.tz_localize('UTC').tz_convert(
        'Europe/Berlin') for t in df['created_at']]
    print(df['created_at_tz'].head(3))
    print(df['created_at'].head(3))

# make sure 'id' is not converted to float
    df['id'] = df['id'].astype(np.int64)

# this does not work
#    df['created_at'] = df['created_at'].astype('datetime64[ns]')
    print(df.info())
    column_order = ['id', 'created_at', 'created_at_tz', 'geo', 'place', 'coordinates',
                    'text', 'text_en', 'truncated', 'name', 'screen_name', ]
    df.to_csv(outfile, index=False, columns=column_order)
    df = pd.read_csv(outfile)
    print(df.info())


# ====================================
# scratch : duplicatino check
# input_file = './data/tweet_bus_de_en.csv'
# df = pd.read_csv(input_file)
# print(len(df))
# print(len(df.drop_duplicates(subset=['id'])))

# ====================================
# END
# ====================================
