
from typing import Iterable, Tuple, List
from random import sample
from itertools import combinations
from numba import njit
import numpy as np
from copy import deepcopy
import os
import pickle


suits = ['s','d','h','c']
values = ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a']


class Card:
    def __init__(self, idx):
        self.idx = idx

    @classmethod
    def from_str(cls, s):
        return Card(values.index(s[0]) * 4 + suits.index(s[1]))

    @classmethod
    def from_value_suit(cls, value, suit):
        return Card(value*4+suit)

    def __str__(self):
        return values[self.idx // 4] + suits[self.idx % 4]

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.idx == other.idx

    def value(self):
        return self.idx//4

    def suit(self):
        return self.idx%4


def str_to_cards(s: str):
    return [Card.from_str(s[i:i+2]) for i in range(0, len(s), 2)]


def str_to_cards_bis(s: str):
    value_str = ""
    cards = []
    for c in s:
        if c in suits:
            value = int(value_str) - 2
            suit = suits.index(c)
            cards.append(Card.from_value_suit(value, suit))
            value_str = ""
        else:
            value_str += c
    return cards


def cards_to_str(cards):
    s = ""
    for card in cards:
        s += str(card)
    return s

def cards_to_str_bis(cards):
    s = ""
    for card in cards:
        value_str = str(card.value()+2)
        suit_str = suits[card.suit()]
        s += value_str + suit_str
    return s


def get_generic_cards(cards, mapping):
    cards = sorted(cards, key=lambda card : card.idx)
    for i, card in enumerate(cards):
        if card.suit() not in mapping:
            mapping[card.suit()] = len(mapping)
        cards[i] = Card.from_value_suit(card.value(), mapping[card.suit()])
    return cards
            

def get_generic_hand(hand, group_sizes=None):
    hand = deepcopy(hand)
    mapping = {}
    if group_sizes is None:
        group_sizes = [2, len(hand)-2]
    new_hand = []
    for group_size in group_sizes:
        if len(new_hand) + group_size > len(hand):
            break
        new_hand += get_generic_cards(hand[len(new_hand):len(new_hand)+group_size], mapping)
    return new_hand

def get_hand_id_table(hand):
    id = 0
    for card in hand:
        id *= 52
        id += card.idx
    return id

def get_hand_id(hand):
    id = 0
    for i in range(7):
        id *= 53
        if i < len(hand):
            id += hand[i].idx+1
    return id

def get_generic_id_table(hand):
    return get_hand_id_table(get_generic_hand(hand, group_sizes=None))

def get_generic_id(hand):
    return get_hand_id(get_generic_hand(hand, group_sizes=[2, 3, 1, 1]))


@njit
def _eval(rank_table: np.ndarray, ref: int, cards: Tuple[int], premature_rank: bool=False):
    p = ref
    for card in cards:
        p = rank_table[p + card + 1]
    if premature_rank:
        p = rank_table[p]
    return p


class Evaluator:
    def __init__(self, precomputed=False):
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
        preflop_table_filename = os.path.join(os.path.dirname(__file__), "preflop_table.npy")
        self.preflop_table = np.load(preflop_table_filename)

        self.precomputed = precomputed
        if precomputed:
            flop_table_filename = os.path.join(os.path.dirname(__file__), "flop_table.pkl")
            turn_table_filename = os.path.join(os.path.dirname(__file__), "turn_table.pkl")
            self.flop_table = pickle.load(open(flop_table_filename, 'rb'))
            self.turn_table = pickle.load(open(turn_table_filename, 'rb'))

    
    def eval(self, cards: Iterable[int]):
        cards = tuple(card.idx for card in cards)
        return _eval(self.rank_table, 53, cards, premature_rank=(len(cards) < 7))

    def _get_contribution(self, cards, idx):
        true_rank = self.eval(cards)
        fake_cards = deepcopy(cards)
        max_contrib = 0
        for card in self.deck:
            if card in cards:
                continue
            fake_cards[idx] = card
            fake_rank = self.eval(fake_cards)
            contribution = true_rank - fake_rank
            if contribution > max_contrib:
                max_contrib = contribution
        return max_contrib

    def get_checker(self, pocket: List[Card], board: List[Card], subrank=False):
        cards = pocket + board
        hand_size = [-1, 1, 2, 4, 3, 5, 5, 5, 4, 5][self.eval(cards) >> 12]
        contributions = [self._get_contribution(cards, idx) for idx in range(len(cards))]
        sorted_cards = sorted(list(range(len(cards))), reverse=True, key=lambda idx: (contributions[idx], idx))
        hand_cards = sorted_cards[:hand_size]
        board_contribution = sum(1 for card in hand_cards if card >= 2)
        return board_contribution / hand_size

    def rank_to_str(self, rank: int):
        return self.rank_to_str_dict[rank >> 12]

    def check_odds_exact(self, pocket: List[Card], board: List[Card]):
        unseen_deck = tuple(card.idx for card in self.deck if card not in (pocket + board))
        pocket = tuple(card.idx for card in pocket)
        board = tuple(card.idx for card in board)
        
        if len(board) > 0:
            board_ref = _eval(self.rank_table, 53, board)
        else:
            board_ref = 53

        board_pocket_ref = _eval(self.rank_table, board_ref, pocket)

        wins = 0
        draws = 0
        total = 0
        for unseen_board in combinations(unseen_deck, 5-len(board)):
            if len(unseen_board) == 0:
                our_strength = board_pocket_ref
                full_board_ref = board_ref
            else:
                our_strength = _eval(self.rank_table, board_pocket_ref, unseen_board)
                full_board_ref = _eval(self.rank_table, board_ref, unseen_board)

            undrawn_deck = tuple(card for card in unseen_deck if card not in unseen_board)
            for opp_cards in combinations(undrawn_deck, 2):
                opp_strength = _eval(self.rank_table, full_board_ref, opp_cards)
                if our_strength > opp_strength:
                    wins += 1
                elif our_strength == opp_strength:
                    draws += 1
                total += 1
        return wins/total, draws/total

    def check_odds_monte_carlo(self, pocket: List[Card], board: List[Card], n_samples: int):
        unseen_deck = tuple(card.idx for card in self.deck if card not in pocket + board)
        pocket = tuple(card.idx for card in pocket)
        board = tuple(card.idx for card in board)
        wins = 0
        draws = 0
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
            elif our_strength == opp_strength:
                draws += 1
            total += 1
        return wins/total, draws/total

    def check_odds_flop(self, pocket: List[Card], board: List[Card]):
        if self.precomputed:
            return self.flop_table[get_generic_id_table(pocket + board)]
        else:
            return self.check_odds_monte_carlo(pocket, board, 20000)

    def check_odds_turn(self, pocket: List[Card], board: List[Card]):
        if self.precomputed:
            return self.turn_table[get_generic_id_table(pocket + board)]
        else:
            return self.check_odds_exact(pocket, board)

    def check_odds_preflop(self, pocket: List[Card]):
        return tuple(self.preflop_table[pocket[0].idx, pocket[1].idx])

    def check_odds(self, pocket: List[Card], board: List[Card]):
        if len(board) == 0:
            return self.check_odds_preflop(pocket)
        if len(board) == 3:
            return self.check_odds_flop(pocket, board)
        if len(board) == 4:
            return self.check_odds_turn(pocket, board)
        if len(board) == 5:
            return self.check_odds_exact(pocket, board)
        raise RuntimeError("invalid board size")

    def full_evaluation(self, pocket_str: str, board_str: str, bis_formatting=False):
        if not bis_formatting:
            pocket = str_to_cards(pocket_str)
            board = str_to_cards(board_str)
        else:
            pocket = str_to_cards_bis(pocket_str)
            board = str_to_cards_bis(board_str)
        if board == []:
            prob_win, prob_draw = self.check_odds(pocket, [])
            return None, None, prob_win, prob_draw
        rank = self.eval(pocket + board)
        checker = self.get_checker(pocket, board)
        prob_win, prob_draw = self.check_odds(pocket, board)
        return rank, checker, prob_win, prob_draw


def run_preflop(evaluator, pocket):
    _, _, prob_win, prob_draw = evaluator.full_evaluation(pocket, "")
    print(f"Prob win : {prob_win*100:.2f}%, Prob draw : {prob_draw*100:.2f}%\n")


def run_after_flop(evaluator, cards):
    rank, checker, prob_win, prob_draw = evaluator.full_evaluation(pocket, cards)
    rank_str = evaluator.rank_to_str(rank)
    print(f"Rank : {rank_str} ({rank}),  Checker : {checker:.2f},  Prob win : {prob_win*100:.2f}%, Prob draw : {prob_draw*100:.2f}%\n")


def run(pocket='', flop='', turn='', river=''):
    evaluator = Evaluator()

    print("")
    if pocket == '':
        pocket = input()
    
    print("Preflop : ")
    run_preflop(evaluator, pocket)

    if flop == '':
        return
    print("Flop")
    run_after_flop(evaluator, flop)

    if turn == '':
        return
    print("Turn")
    run_after_flop(evaluator, flop + turn)

    if river == '':
        return
    print("River")
    run_after_flop(evaluator, flop + turn + river)


if __name__ == '__main__':
    pocket = '2sqs'
    flop = 'kdjdjh'
    turn = 'js'
    river = 'jc'
    run(pocket, flop, turn, river)