
import numpy as np
import pandas as pd
from poker_eval import get_generic_hand, get_hand_id, str_to_cards_bis, cards_to_str_bis, Evaluator
import multiprocessing
import os


def process_row_generic(row):
    hand = []
    for group in ['Pocket', 'Table', 'Turn', 'River']:
        if row[group] != "-":
            hand += str_to_cards_bis(row[group])
    generic_hand = get_generic_hand(hand, group_sizes=[2, 3, 1, 1])
    generic_id = get_hand_id(generic_hand)
    row['generic_id'] = generic_id
    generic_pocket = generic_hand[:2]
    row['generic_pocket'] = cards_to_str_bis(generic_pocket)
    if len(generic_hand) > 2:
        generic_table = generic_hand[2:5]
        row['generic_table'] = cards_to_str_bis(generic_table)
    if len(generic_hand) > 5:
        generic_turn = generic_hand[5:6]
        row['generic_turn'] = cards_to_str_bis(generic_turn)
    if len(generic_hand) > 6:
        generic_river = generic_hand[6:7]
        row['generic_river'] = cards_to_str_bis(generic_river)
    return row


def process_chunk_generic(df, idx, queue):
    df['generic_id'] = -1
    df['generic_table'] = "-"
    df['generic_turn'] = "-"
    df['generic_river'] = "-"
    df['generic_pocket'] = "-"
    df = df.apply(process_row_generic, axis=1)
    queue.put((idx, df))


def process_row_proba_stage(evaluator: Evaluator, pocket_str: str, board_str: str, row, stage):
    rank, checker, prob_win, prob_draw = evaluator.full_evaluation(pocket_str, board_str, bis_formatting=True)
    row['proba_win_' + stage] = prob_win
    row['proba_draw_' + stage] = prob_draw
    if stage != 'preflop':
        try:
            rank_str = evaluator.rank_to_str(rank)
        except TypeError:
            print(pocket_str, board_str, row, stage)
            return
        row['best_hand_' + stage] = rank_str
        row['checker_' + stage] = checker


def process_row_proba(evaluator, row):
    process_row_proba_stage(evaluator, row['Pocket'], "", row, 'preflop')
    if row['Table'] == "-":
        return row
    process_row_proba_stage(evaluator, row['Pocket'], row['Table'], row, 'flop')
    if row['Turn'] == "-":
        return row
    process_row_proba_stage(evaluator, row['Pocket'], row['Table'] + row['Turn'], row, 'turn')
    if row['River'] == "-":
        return row
    process_row_proba_stage(evaluator, row['Pocket'], row['Table'] + row['Turn'] + row['River'], row, 'river')
    return row


def process_chunk_proba(df, idx, queue):
    evaluator = Evaluator(precomputed=True)
    df['proba_win_preflop'] = np.nan
    df['proba_draw_preflop'] = np.nan
    for stage in ('flop', 'turn', 'river'):
        df['best_hand_' + stage] = "-"
        for col in ('checker', 'proba_win', 'proba_draw'):
            df[col + '_' + stage] = np.nan
    df = df.apply(lambda row : process_row_proba(evaluator, row), axis=1)
    queue.put((idx, df))
    

def process_parallel_chunk(df_iterator, output, first, n_workers, process_func) -> bool:
    queue = multiprocessing.Queue()
    processes = []
    for i in range(n_workers):
        try:
            chunk = next(df_iterator)
            p = multiprocessing.Process(target=process_func, args=(chunk, i, queue))
            processes.append(p)
        except StopIteration:
            break
    
    n_processes = len(processes)
    for p in processes:
        p.start()

    results = []
    while len(results) < n_processes:
        results.append(queue.get())
    results = sorted(results, key=lambda x : x[0])
    for _, df in results:
        if first:
            df.to_csv(output, index=False)
        else:
            df.to_csv(output, index=False, mode='a', header=False)
        first = False

    return n_processes < n_workers
    

def process_csv(filename, output, process_chunk_func, n_workers, chunksize):
    df_iterator = pd.read_csv(filename, keep_default_na=False, chunksize=chunksize, iterator=True)
    n_processed = 0
    while not process_parallel_chunk(df_iterator, output, n_processed==0, n_workers, process_chunk_func):
        n_processed += n_workers*chunksize
        print('\r{:,}'.format(n_processed), end='')
    print("")


def filter_df(df):
    return df.drop_duplicates(subset=['generic_id'], keep='first')


def filter_csv(filename, output, chunksize):
    first = True
    n_processed = 0
    for df in pd.read_csv(filename, keep_default_na=False, chunksize=chunksize, iterator = True):
        result = filter_df(df)
        if first:
            result.to_csv(output, index=False)
            first = False
        else:
            result.to_csv(output, index=False, mode='a', header=False)
        n_processed += chunksize
        print('\r{:,}'.format(n_processed), end='')
    print("")
    df = pd.read_csv(output, keep_default_na=False)
    df = filter_df(df)
    df.to_csv(output, index=False)
    

def run(csvpath, output_generic, output_filtered, output_with_probs):
    print("Generic hands processing")
    process_csv(csvpath, output_generic, process_chunk_generic, n_workers=8, chunksize=100000)
    print("Filtering duplicates")
    filter_csv(output_generic, output_filtered, chunksize=1000000)
    print("Odds processing")
    process_csv(output_filtered, output_with_probs, process_chunk_proba, n_workers=2, chunksize=100000)
    #os.remove(output_generic)
    #os.remove(output_filtered) 

if __name__ == '__main__':
    run(
        input='dbg_260123.csv', 
        output_generic='dbg_260123_generic.csv',
        output_filtered='dbg_260123_filtered.csv',
        output_with_probs='dbg_260123_filtered_with_probs.csv'
        )
