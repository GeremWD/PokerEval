
import pandas as pd
from poker_eval import str_to_cards_bis, get_generic_id


def run(pocket, flop, turn, river, dbpath):
    hand = str_to_cards_bis(pocket + flop + turn + river)
    generic_id = get_generic_id(hand)
    db = pd.read_csv(dbpath)
    print(db[db['generic_id'] == generic_id].iloc[0])


if __name__ == '__main__':
    pocket = '3d11c'
    flop = '6c12s10h'
    turn = ''
    river = ''
    run(pocket, flop, turn, river, dbpath='dbg_260123_filtered_with_probs.csv')