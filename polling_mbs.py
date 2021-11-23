
import sys
import config
from mbs.mbs import *
import pretty_errors
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
