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

    MKL_AK = config.MONKEYLEARN_API_KEY
    MKL_ST_MODEL_ID = config.MONKEYLEARN_SENTIMENT_MODEL_ID
    MKL_EX_MODEL_ID = config.MONKEYLEARN_KEYWORD_EXTRACTOR_MODEL_ID

    DATA_DIR = Path('./data')
    LOG_FILE = Path('./log/log_file.txt')
    LOG_FILE_COLOR = Path('./log/log_file_color.txt')
#    NOTE_FILE = Path('./note/summary.yaml')

#    freq = '12H'
    polling_tweets(
        BT, DEEPL_AK,
        outfile=outfile,
        n_stopper=n_stopper, t_sleep=t_sleep,
        MKL_AK=MKL_AK,
        MKL_ST_MODEL_ID=MKL_ST_MODEL_ID,
        MKL_EX_MODEL_ID=MKL_EX_MODEL_ID,
        DATA_DIR=DATA_DIR,
        LOG_FILE=LOG_FILE,
        LOG_FILE_COLOR=LOG_FILE_COLOR)

# =========
# scratch
# =========
