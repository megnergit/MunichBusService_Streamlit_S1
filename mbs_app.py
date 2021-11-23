import yaml
import time
import urllib
import os
from pathlib import Path
import plotly
import plotly.express as px
import json
from os import P_NOWAIT
from requests_oauthlib import OAuth1Session
import requests
from geopandas.tools import geocode
import re
import deepl
import tweepy
from monkeylearn import MonkeyLearn
import pandas as pd
import numpy as np
import typing

import plotly.graph_objs as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from plotly.offline import plot

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import folium
from folium import Marker, GeoJson, Circle
from folium.plugins import HeatMap
import geopandas as gpd

import webbrowser
from datetime import datetime
import importlib
import config
import sys
from mbs.mbs import *
import streamlit as st
import folium

from streamlit_autorefresh import st_autorefresh

# ====================================
# Authentication
# ====================================

AK = config.API_KEY  # not really necessary
AKS = config.API_KEY_SECRET  # not really necessary
BT = config.BEARER_TOKEN

MKL_AK = config.MONKEYLEARN_API_KEY
MKL_ST_MODEL_ID = config.MONKEYLEARN_SENTIMENT_MODEL_ID
MKL_EX_MODEL_ID = config.MONKEYLEARN_KEYWORD_EXTRACTOR_MODEL_ID

DATA_DIR = Path('./data')
LOG_FILE = Path('./log/log_file.txt')
LOG_FILE_COLOR = Path('./log/log_file_color.txt')
NOTE_FILE = Path('./note/summary.yaml')

DEEPL_AK = config.DEEPL_API_KEY

size = 640
input_file = DATA_DIR/'tweet_bus_de_en.csv'
n_last = 30
# center = (11.57540, 48.13714)
# ============================================
# scratch streamlit
# --------------------------------------------
usecols = ['id', 'created_at', 'geo', 'place', 'coordinates', 'text',
           'text_en', 'truncated', 'name', 'screen_name']
df = pd.read_csv(input_file,
                 parse_dates=['created_at'],
                 usecols=usecols)

# =============================================
# monkey learn
# --------------------------------------------
df_stx = get_mkl_st_dummy(df, MKL_AK)
df_kex = get_mkl_ex_dummy(df_stx, MKL_AK)
df_kex.to_csv(DATA_DIR/'mbs_kex.csv', index=False)
df_geo = extract_place(df_kex)
df_geo.to_csv(DATA_DIR/'mbs_geo.csv', index=False)

df_pn = add_sentiment_digit(df_kex)
df_agg = aggregate_sentiment(df_pn, freq='12H')
# '120S' '2H' '1D'
df_agg.to_csv(DATA_DIR/'mbs_agg.csv', index=False)

# --------------------------------------------
# calculate daily aggregate
# --------------------------------------------

fig_agg = visualize_agg(df_agg, size)
fig_count = visualize_count(df_agg, size)
# --------------------------------------------
fig_pn = visualize_pn(df_pn, size, vertical=True)
# --------------------------------------------
# wordcloud
# --------------------------------------------
wc = create_wordcloud(df_kex, size)
fig_wc = visualize_wc(wc)
# --------------------------------------------
# folium
# --------------------------------------------
m_1 = plot_sentiment(df_kex)

# --------------------------------------------
# logfile
# --------------------------------------------
i = 0
with open(LOG_FILE, 'r') as f:
    log_text = f.readlines()
    print(i+1)

log_text = [s.replace('\n', '') for s in log_text]

with open(LOG_FILE_COLOR, 'r') as f:
    log_text_color = f.readlines()

log_text_color = [s.replace('\n', '<br />') for s in log_text_color]

# --------------------------------------------
# read YAML markdown text

with open(NOTE_FILE, 'r') as s:
    try:
        note = yaml.safe_load(s)
    except yaml.YAMLError as e:
        print(e)

# =====================================================================
# streamlit scratch
# =====================================================================
st.set_page_config(layout='wide')
count = st_autorefresh(interval=1000 * 1200, limit=16, key="sagasitemiyou")

# --------------------------------------------
# 1. row : cautions
# --------------------------------------------
col1, col2, col3 = st.columns((0.9, 0.9, 0.9))
with col1:
    st.markdown(note['note1'], unsafe_allow_html=True)

with col2:
    st.markdown(note['note2'], unsafe_allow_html=True)

with col3:
    st.markdown(note['note3'], unsafe_allow_html=True)

st.markdown("""___""")

# --------------------------------------------
# 2. row questions and conclusions
# --------------------------------------------
st.title('How People like Munich Bus Service')
col1, col2, col3 = st.columns([0.9, 0.1, 1.4])
with col1:
    st.markdown(note['questions'], unsafe_allow_html=True)

with col3:
    st.markdown(note['conclusions'], unsafe_allow_html=True)

# --------------------------------------------
# 3. row
# --------------------------------------------
st.markdown('### Overall Sentiment')
st.plotly_chart(fig_agg, use_container_width=True)
st.markdown('### How many Tweets about Bus?')
st.plotly_chart(fig_count, use_container_width=True)

# --------------------------------------------
# 5. row : report and polling log
# --------------------------------------------
col1, col2 = st.columns((1, 0.9))
log = '\n'.join(log_text)
log_color = ' '.join(log_text_color[-4:])
with col1:
    st.markdown(note['map_caption'])

with col2:
    st.markdown('### Polling log')
    st.markdown(log_color, unsafe_allow_html=True)

# --------------------------------------------
# 4. row
# --------------------------------------------
df_words = pd.DataFrame(dict(word=wc.words_.keys(), frac=wc.words_.values()))
df_words.sort_values(['frac'], ascending=False, inplace=True)

col1, col2, col3 = st.columns((2, 0.9, 0.9))
with col1:
    st.markdown('### Where people are satisfied/dissatified?')
    m_1.to_streamlit(height=size*1)

with col2:
    text = f'### Last {n_last} Tweets'
    st.markdown(text)
    st.plotly_chart(fig_pn, use_container_width=True)

with col3:
    st.markdown('### Satisfied/dissatified with...')
    st.image(wc.to_image())
    st.dataframe(df_words)


# --------------------------------------------
#  6. row
# --------------------------------------------
st.markdown('### All Data')
st.dataframe(df_kex.drop(
    ['Unnamed: 0', 'name', 'screen_name'], axis=1, errors='ignore'),
    height=size)

# --------------------------------------------
