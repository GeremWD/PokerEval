
import time
import numpy as np
import pandas as pd
from poker import Evaluator, Card
from multiprocessing import Process, Queue

n_samples = 20_000
n_workers = 8

def str_to_cards(s: str):
    return [Card.from_str(s[i:i+2]) for i in range(0, len(s), 2)]


def process_stage(evaluator: Evaluator, pocket_str: str, board_str: str, row, stage):
    pocket = str_to_cards(pocket_str)
    board = str_to_cards(board_str)
    if stage == 'preflop':
        row['proba_preflop'] = evaluator.check_odds(pocket, [])
        return
    rank = evaluator.eval(pocket + board)
    rank_str = evaluator.rank_to_str(rank)
    row['proba_' + stage] = evaluator.check_odds(pocket, board, n_samples)
    row['best_hand_' + stage] = rank_str
    row['checker_' + stage] = evaluator.get_checker(pocket, board)


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


def process_thread(rows, start_row_idx, queue):
    evaluator = Evaluator()
    for idx, row in enumerate(rows):
        process_row(evaluator, row)
        queue.put((start_row_idx + idx, row))


def process_csv(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    df = df.fillna('')
    df['proba_preflop'] = np.nan
    for stage in ('flop', 'turn', 'river'):
        for col in ('best_hand', 'checker', 'proba'):
            df[col + '_' + stage] = np.nan

    rows = [row for _, row in df.iterrows()]
    queue = Queue()

    n_rows = len(rows)

    start = time.time()
    for worker_idx in range(n_workers):
        start_row_idx = worker_idx * (n_rows//n_workers)
        end_row_idx = (worker_idx + 1) * (n_rows//n_workers)
        if worker_idx == n_workers-1:
            end_row_idx = n_rows
        worker_rows = [row.copy() for row in rows[start_row_idx:end_row_idx]]
        worker = Process(target=process_thread, args=(worker_rows, start_row_idx, queue))
        worker.start()

    n_processed = 0
    while n_processed < n_rows:
        idx, row = queue.get()
        df.iloc[idx] = row
        n_processed += 1
        if n_processed % 100 == 0:
            end = time.time()
            print(f"  Time spent : {end - start:.1f} seconds | Rows processed : {n_processed}", end='\r')
    print('')

    df.to_csv('output.csv', sep=';', index=False)


if __name__ == '__main__':
    process_csv('input.csv')