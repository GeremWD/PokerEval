
from poker_eval import Evaluator, Card, suits, values
import numpy as np

preflop_table = np.zeros((52, 52), dtype=float)
evaluator = Evaluator(preflop_table=False)

n_samples = 100_000

for value_idx_1 in range(13):
    for value_idx_2 in range(value_idx_1, 13):
        print(value_idx_1, value_idx_2)
        value1 = values[value_idx_1]
        value2 = values[value_idx_2]
        diff_color_cards = [Card.from_str(value1 + suits[0]), Card.from_str(value2 + suits[1])]
        diff_color_odds = evaluator.check_odds(diff_color_cards, [], n_samples)

        if value1 != value2:
            same_color_cards = [Card.from_str(value1 + suits[0]), Card.from_str(value2 + suits[0])]
            same_color_odds = evaluator.check_odds(same_color_cards, [], n_samples)

        for suit1 in suits:
            for suit2 in suits:
                if suit1 == suit2 and value1 == value2:
                    continue
                card1 = Card.from_str(value1 + suit1)
                card2 = Card.from_str(value2 + suit2)
                if suit1 == suit2:
                    odds = same_color_odds
                else:
                    odds = diff_color_odds
                preflop_table[card1.idx, card2.idx] = odds
                preflop_table[card2.idx, card1.idx] = odds

np.save('preflop_table.npy', preflop_table)