
import time
import numpy as np
import pandas as pd
from poker_eval import Evaluator

def process_stage(evaluator: Evaluator, pocket_str: str, board_str: str, row, stage):
    rank, checker, prob_win, prob_draw = evaluator.full_evaluation(pocket_str, board_str)
    row['proba_win_' + stage] = prob_win
    row['proba_draw_' + stage] = prob_draw
    if stage != 'preflop':
        rank_str = evaluator.rank_to_str(rank)
        row['best_hand_' + stage] = rank_str
        row['checker_' + stage] = checker

def process_row(evaluator: Evaluator, row):
    process_stage(evaluator, row['Pocket'], "", row, 'preflop')
    if row['Table'] == "":
        return
    process_stage(evaluator, row['Pocket'], row['Table'], row, 'flop')
    if row['Turn'] == "":
        return
    process_stage(evaluator, row['Pocket'], row['Table'] + row['Turn'], row, 'turn')
    if row['River'] == "":
        return
    process_stage(evaluator, row['Pocket'], row['Table'] + row['Turn'] + row['River'], row, 'river')

def process_csv(csv_path, output_path, sep=';'):
    df = pd.read_csv(csv_path, sep=sep)
    df = df.fillna('')
    df['proba_win_preflop'] = np.nan
    df['proba_draw_preflop'] = np.nan
    for stage in ('flop', 'turn', 'river'):
        for col in ('best_hand', 'checker', 'proba_win', 'proba_draw'):
            df[col + '_' + stage] = np.nan

    evaluator = Evaluator(precomputed=True)
    start = time.time()
    n_processed = 0
    for idx, row in df.iterrows():
        process_row(evaluator, row)
        df.iloc[idx] = row
        n_processed += 1
        if n_processed % 100 == 0:
            end = time.time()
            print(f"  Time spent : {end - start:.1f} seconds | Rows processed : {n_processed}", end='\r')
    print('')
    df.to_csv(output_path, sep=sep, index=False)


if __name__=='__main__':
    process_csv(
        csv_path="input.csv",
        output_path="output.csv",
    )