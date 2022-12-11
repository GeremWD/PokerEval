import poker_eval

evaluator = poker_eval.Evaluator()

def print_result_preflop(odds):
    print(f"Odds : {odds*100:.2f} %\n")

def print_result_after_flop(rank, checker, odds):
    rank_str = evaluator.rank_to_str(rank)
    print(f"Rank : {rank_str} ({rank}),  Checker : {checker:.2f},  Odds : {odds*100:.2f} %\n")

pocket = 'ad8h'
flop = '4cth2d'
turn = 'as'
river = 'td'

_, _, odds = evaluator.full_evaluation(pocket, "")
print("Preflop : ")
print_result_preflop(odds)
print("Flop")
print_result_after_flop(*evaluator.full_evaluation(pocket, flop, n_samples=20000))
print("Turn")
print_result_after_flop(*evaluator.full_evaluation(pocket, flop + turn))
print("River")
print_result_after_flop(*evaluator.full_evaluation(pocket, flop + turn + river))