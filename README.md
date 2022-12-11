# PokerEval

## Usage :

To evaluate one situation, i.e. obtain the hand rank, the checker and the odds :
```py
import poker_eval
evaluator = poker_eval.Evaluator()
rank, checker, odds = evaluator.full_evaluation(pocket='ad8h', board='4cth2d', n_samples=20000)
```

If the board is empty (preflop), rank and checker will be None.

The n_samples parameter is the number of simulations performed to approximate the odds.
It is only used for the flop. 
For the preflop, the odds are precomputed. For the turn and river, they are computed exactly.

You should avoid recreating the evaluator, because it allocates a big table in memory during initialization.

The rank is represented by an integer but can be converted to a human readable format with :
```py
readable_rank = evaluator.rank_to_str(rank)
```
The result belongs to
```py
('HIGH_CARD', 'ONE_PAIR', 'TWO_PAIR', 'THREE_OF_A_KIND', 'STRAIGHT', 'FLUSH', 'FULL_HOUSE', 'FOUR_OF_A_KIND', 'STRAIGHT_FLUSH')
```

To process an entire csv file :

```py
import poker_eval

if __name__ == '__main__': # Important guard to prevent some multiprocessing issues
    poker_eval.process_csv(
        csv_path="input.csv",
        output_path="output.csv",
        n_samples=20000,
        n_workers=8
    )
```
