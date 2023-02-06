
from fastapi import FastAPI
import pandas as pd
from poker_eval import str_to_cards_bis, get_generic_id

dbpath = "dbg_260123_filtered_with_probs.csv"
db = pd.read_csv(dbpath)
db = db.drop(['generic_table', 'generic_turn', 'generic_river', 'generic_pocket'], axis=1)

app = FastAPI()

@app.get("/lookup/{pocket}/{flop}/{turn}/{river}")
def lookup(pocket, flop, turn, river):
    if pocket == "-":
        pocket = ""
    if flop == "-":
        flop = ""
    if turn == "-":
        turn = ""
    if river == "-":
        river = ""
    hand = str_to_cards_bis(pocket + flop + turn + river)
    generic_id = get_generic_id(hand)
    result = db[db['generic_id'] == generic_id].iloc[0]
    result = result.fillna('')
    return result.to_dict()