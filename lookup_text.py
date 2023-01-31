
import pandas as pd
from poker_eval import str_to_cards_bis, get_generic_id
import json
import time
import os

clear = lambda: os.system('cls')

def run(dbpath):
    db = pd.read_csv(dbpath)
    db = db.drop(['generic_table', 'generic_turn', 'generic_river', 'generic_pocket'], axis=1)
    while True:
        hand = json.loads(open('hand.json').read())
        try:
            hand = str_to_cards_bis(hand['pocket'] + hand['flop'] + hand['turn'] + hand['river'])
            generic_id = get_generic_id(hand)
            clear()
            print(db[db['generic_id'] == generic_id].iloc[0])
        except KeyboardInterrupt:
            return
        except:
            pass
        time.sleep(0.05)

if __name__ == '__main__':
    run(dbpath='dbg_260123_filtered_with_probs.csv')