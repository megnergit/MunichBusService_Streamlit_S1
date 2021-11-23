
import sys
# import time
# import urllib
# import os
# from pathlib import Path
# import plotly
# import plotly.express as px
# import json
import config
# from requests_oauthlib import OAuth1Session
# import requests
# from geopandas.tools import geocode
# import re
# import deepl
# # import tweepy
# from monkeylearn import MonkeyLearn
# import pandas as pd
# import numpy as np
# import typing

import plotly.graph_objs as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from plotly.offline import plot

# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# import folium
# from folium import Marker, GeoJson, Circle
# from folium.plugins import HeatMap
# import geopandas as gpd

#from datetime import datetime
import pretty_errors
from mbs.mbs import *
# ===============================================
# polling tweets
# ===============================================
if __name__ == '__main__':
    args = sys.argv

    if len(args) == 4:
        outfile = args[1]
        n_stopper = int(args[2])
        t_sleep = int(args[3])

    else:
        print(f'\033[33mUsage : \033[96mpython3 \033[0mpolling_mbs.py \
[outfile="tweet.csv"] [n_stopper=100] [t_sleep=1]')
        exit()

    AK = config.API_KEY  # not really necessary
    AKS = config.API_KEY_SECRET  # not really necessary
    BT = config.BEARER_TOKEN
    DEEPL_AK = config.DEEPL_API_KEY

    polling_tweet_mbs(BT, DEEPL_AK, outfile, n_stopper, t_sleep)
