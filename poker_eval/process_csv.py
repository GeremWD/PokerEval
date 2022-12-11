
import time
import numpy as np
import pandas as pd
from poker_eval import Evaluator
from multiprocessing import Process, Queue


def process_stage(evaluator: Evaluator, pocket_str: str, board_str: str, row, stage, n_samples):
    rank, checker, odds = evaluator.full_evaluation(pocket_str, board_str, n_samples)
    row['proba_' + stage] = odds
    if stage != 'preflop':
        rank_str = evaluator.rank_to_str(rank)
        row['best_hand_' + stage] = rank_str
        row['checker_' + stage] = checker


def process_row(evaluator: Evaluator, row, n_samples):
    process_stage(evaluator, row['Pocket'], "", row, 'preflop', n_samples)
    if row['Table'] == "":
        return
    process_stage(evaluator, row['Pocket'], row['Table'], row, 'flop', n_samples)
    if row['Turn'] == "":
        return
    process_stage(evaluator, row['Pocket'], row['Table'] + row['Turn'], row, 'turn', n_samples)
    if row['River'] == "":
        return
    process_stage(evaluator, row['Pocket'], row['Table'] + row['Turn'] + row['River'], row, 'river', n_samples)


def process_thread(rows, start_row_idx, queue, n_samples):
    evaluator = Evaluator()
    for idx, row in enumerate(rows):
        process_row(evaluator, row, n_samples)
        queue.put((start_row_idx + idx, row))


def process_csv(csv_path, output_path, n_samples = 20_000, n_workers = 8):
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
        worker = Process(target=process_thread, args=(worker_rows, start_row_idx, queue, n_samples))
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

    df.to_csv(output_path, sep=';', index=False)
