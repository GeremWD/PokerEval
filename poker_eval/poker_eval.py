
from typing import Iterable, Tuple, List
from random import sample
from itertools import combinations
from numba import njit
import numpy as np
from copy import deepcopy
import os
import argparse


suits = ['s','d','h','c']
values = ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a']


class Card:
    def __init__(self, idx):
        self.idx = idx

    @classmethod
    def from_str(cls, s):
        return Card(values.index(s[0]) * 4 + suits.index(s[1]))

    def __str__(self):
        return values[self.idx // 4] + suits[self.idx % 4]

    def __eq__(self, other):
        return self.idx == other.idx

    def value(self):
        return self.idx//4


def str_to_cards(s: str):
    return [Card.from_str(s[i:i+2]) for i in range(0, len(s), 2)]


@njit
def _eval(rank_table: np.ndarray, ref: int, cards: Tuple[int], premature_rank: bool=False):
    p = ref
    for card in cards:
        p = rank_table[p + card + 1]
    if premature_rank:
        p = rank_table[p]
    return p

class Evaluator:
    def __init__(self, preflop_table=True):
        rank_table_filename = os.path.join(os.path.dirname(__file__), "rank_table.bin")
        rank_table_file = open(rank_table_filename, "r")
        self.rank_table = np.fromfile(rank_table_file, dtype=np.int32)
        self.rank_to_str_dict = {
            1: 'HIGH_CARD',
            2: 'ONE_PAIR',
            3: 'TWO_PAIR',
            4: 'THREE_OF_A_KIND',
            5: 'STRAIGHT',
            6: 'FLUSH',
            7: 'FULL_HOUSE',
            8: 'FOUR_OF_A_KIND',
            9: 'STRAIGHT_FLUSH'  
        }
        self.deck = [Card(i) for i in range(52)]
        if preflop_table:
            preflop_table_filename = os.path.join(os.path.dirname(__file__), "preflop_table.npy")
            self.preflop_table = np.load(preflop_table_filename)
        else:
            self.preflop_table = None
    
    def eval(self, cards: Iterable[int]):
        cards = tuple(card.idx for card in cards)
        return _eval(self.rank_table, 53, cards, premature_rank=(len(cards) < 7))

    def _does_card_contribute(self, cards, idx, subrank=False):
        true_rank = self.eval(cards)
        if true_rank >> 12 == 1:
            return cards[idx].value() == max(card.value() for card in cards)

        fake_cards = deepcopy(cards)
        for card in self.deck:
            if card in cards:
                continue
            fake_cards[idx] = card
            fake_rank = self.eval(fake_cards)
            if ((not subrank) and (fake_rank >> 12 < true_rank >> 12)) or (subrank and (fake_rank < true_rank)):
                return True
        return False

    def get_checker(self, pocket: List[Card], board: List[Card], subrank=False):
        cards = pocket + board
        pocket_contribution = 0
        board_contribution = 0
        for idx in range(len(cards)): 
            if self._does_card_contribute(cards, idx, subrank):
                if idx < 2:
                    pocket_contribution += 1
                else:
                    board_contribution += 1
        if board_contribution + pocket_contribution == 0:
            return self.get_checker(pocket, board, True)
        return board_contribution / (board_contribution + pocket_contribution)

    def rank_to_str(self, rank: int):
        return self.rank_to_str_dict[rank >> 12]

    def check_odds_exact(self, pocket: List[Card], board: List[Card]):
        unseen_deck = tuple(card.idx for card in self.deck if card not in pocket + board)
        pocket = tuple(card.idx for card in pocket)
        board = tuple(card.idx for card in board)
        
        if len(board) > 0:
            board_ref = _eval(self.rank_table, 53, board)
        else:
            board_ref = 53

        board_pocket_ref = _eval(self.rank_table, board_ref, pocket)

        wins = 0
        total = 0
        for unseen_board in combinations(unseen_deck, 5-len(board)):
            if len(unseen_board) == 0:
                our_strength = board_pocket_ref
                full_board_ref = board_ref
            else:
                our_strength = _eval(self.rank_table, board_pocket_ref, unseen_board, 
                    premature_rank=(len(board) < 5)
                )
                full_board_ref = _eval(self.rank_table, board_ref, unseen_board)

            undrawn_deck = tuple(card for card in unseen_deck if card not in unseen_board)
            for opp_cards in combinations(undrawn_deck, 2):
                opp_strength = _eval(self.rank_table, full_board_ref, opp_cards,
                    premature_rank=(len(board) < 5)
                )
                if our_strength > opp_strength:
                    wins += 1
                total += 1
        return wins/total

    def check_odds_monte_carlo(self, pocket: List[Card], board: List[Card], n_samples: int):
        unseen_deck = tuple(card.idx for card in self.deck if card not in pocket + board)
        pocket = tuple(card.idx for card in pocket)
        board = tuple(card.idx for card in board)

        wins = 0
        total = 0
        for _ in range(n_samples):
            board_opp_cards = tuple(sample(unseen_deck, 5-len(board)+2))
            opp_cards = board_opp_cards[:2]
            unseen_board = board_opp_cards[2:]
            full_board = board + unseen_board
            our_strength = _eval(self.rank_table, 53, full_board + pocket)
            opp_strength = _eval(self.rank_table, 53, full_board + opp_cards)
            if our_strength > opp_strength:
                wins += 1
            total += 1
        return wins/total

    def check_odds_preflop(self, pocket: List[Card]):
        return self.preflop_table[pocket[0].idx, pocket[1].idx]

    def check_odds(self, pocket: List[Card], board: List[Card], n_samples: int=10000):
        if len(board) == 0 and self.preflop_table is not None:
            return self.check_odds_preflop(pocket)
        if len(board) >= 4:
            return self.check_odds_exact(pocket, board)
        return self.check_odds_monte_carlo(pocket, board, n_samples)

    def full_evaluation(self, pocket_str, board_str, n_samples=100_000):
        pocket = str_to_cards(pocket_str)
        board = str_to_cards(board_str)
        if board == []:
            return None, None, self.check_odds(pocket, [])
        rank = self.eval(pocket + board)
        checker = self.get_checker(pocket, board)
        odds = self.check_odds(pocket, board, n_samples)
        return rank, checker, odds


def run_preflop(evaluator, pocket):
    _, _, odds = evaluator.full_evaluation(pocket, "")
    print(f"Odds : {odds*100:.2f} %\n")


def run_after_flop(evaluator, cards, n_samples):
    rank, checker, odds = evaluator.full_evaluation(pocket, cards, n_samples)
    rank_str = evaluator.rank_to_str(rank)
    print(f"Rank : {rank_str} ({rank}),  Checker : {checker:.2f},  Odds : {odds*100:.2f} %\n")


def run(pocket='', flop='', turn='', river='', n_samples=20000):
    evaluator = Evaluator()

    print("")
    if pocket == '':
        pocket = input()
    
    print("Preflop : ")
    run_preflop(evaluator, pocket)

    if flop == '':
        return
    print("Flop")
    run_after_flop(evaluator, flop, n_samples)

    if turn == '':
        return
    print("Turn")
    run_after_flop(evaluator, flop + turn, n_samples)

    if river == '':
        return
    print("River")
    run_after_flop(evaluator, flop + turn + river, n_samples)


if __name__ == '__main__':
    pocket = 'ad8h'
    flop = '4cth2d'
    turn = 'as'
    river = ''
    run(pocket, flop, turn, river)