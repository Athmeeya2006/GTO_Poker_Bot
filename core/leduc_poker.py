# core/leduc_poker.py
"""
Leduc Poker:
- 6 cards: two each of J(1), Q(2), K(3)
- 2 players, ante=1 each
- Round 1: bet size 2, max 1 raise
- Flop: community card revealed
- Round 2: bet size 4, max 1 raise
- Showdown: pair with community card beats no pair; higher card breaks ties
"""
from itertools import permutations

# Cards
J, Q, K = 1, 2, 3
DECK = [J, J, Q, Q, K, K]

def get_all_deals():
    """All possible private card deals (p1_card, p2_card, community_card)."""
    deals = []
    for i, c1 in enumerate(DECK):
        for j, c2 in enumerate(DECK):
            if i == j:
                continue
            for k, cc in enumerate(DECK):
                if k == i or k == j:
                    continue
                deals.append((c1, c2, cc))
    return deals

def is_terminal_leduc(history):
    """
    History is a string. '/' separates rounds.
    Round actions: c=check/call, b=bet, f=fold, r=raise
    Terminal if: fold, or both rounds complete.
    """
    if "f" in history:
        return True
    parts = history.split("/")
    if len(parts) < 2:
        return False
    r2 = parts[1]
    # Round 2 ends at: cc, bc, rc, cbc, cbf, crbc, etc.
    # Simplified: ends when second player has acted on a terminal sequence
    return _is_round_over(r2)

def _is_round_over(round_hist):
    if not round_hist:
        return False
    if round_hist.endswith("f"):
        return True
    if round_hist in ["cc", "bc", "rc", "cbc", "cbf", "rbc", "rbf",
                      "crbc", "crbf", "brbc", "brbf"]:
        return True
    return False

def get_actions_leduc(history):
    """Returns valid actions given current history."""
    if is_terminal_leduc(history):
        return []

    parts   = history.split("/")
    round_n = len(parts) - 1
    r_hist  = parts[-1]

    # Count bets/raises in this round
    n_bets = r_hist.count("b") + r_hist.count("r")

    if r_hist == "":
        return ["c", "b"]       # first to act: check or bet
    if r_hist == "c":
        return ["c", "b"]       # second to act after check
    if r_hist == "b":
        if n_bets < 2:          # can raise
            return ["c", "f", "r"]
        return ["c", "f"]       # only call or fold if max raises hit
    if r_hist.endswith("r"):
        if n_bets < 2:
            return ["c", "f", "r"]
        return ["c", "f"]
    if r_hist == "cb":
        return ["c", "f"]       # facing bet after check: call or fold only

    return []

def current_player_leduc(history):
    parts  = history.split("/")
    r_hist = parts[-1]
    n_acts = len(r_hist)
    return n_acts % 2  # alternates: P0 first in each round

def hand_rank(private, community):
    """Higher = better hand."""
    if private == community:
        return 100 + private     # pair: higher pair wins
    return private               # no pair: just card rank

def get_payoff_leduc(history, cards, player):
    """
    cards = [p1_card, p2_card, community_card]
    Returns payoff for player (0 or 1).
    """
    p0_card, p1_card, comm = cards

    # Count pot
    parts = history.split("/")
    r1, r2 = parts[0], parts[1] if len(parts) > 1 else ""

    pot = 2  # antes
    pot += _pot_contribution(r1, bet_size=2)
    pot += _pot_contribution(r2, bet_size=4)

    # Who wins
    if "f" in history:
        # Find who folded
        folder = _find_folder(history)
        winner = 1 - folder
    else:
        rank0 = hand_rank(p0_card, comm)
        rank1 = hand_rank(p1_card, comm)
        if rank0 > rank1:
            winner = 0
        elif rank1 > rank0:
            winner = 1
        else:
            # Tie: split pot
            return 0

    player_contrib = _player_contribution(history, player)
    opponent_contrib = pot - player_contrib

    if winner == player:
        return opponent_contrib
    else:
        return -player_contrib

def _pot_contribution(round_hist, bet_size):
    """Total chips added to pot in this round."""
    total = 0
    for ch in round_hist:
        if ch in ("b", "r", "c"):  # c after a bet = call
            total += bet_size
    return total

def _find_folder(history):
    """Returns player index who folded."""
    parts  = history.split("/")
    n_acts = 0
    for part in parts:
        for ch in part:
            if ch == "f":
                return n_acts % 2
            n_acts += 1
    return -1

def _player_contribution(history, player):
    """How many chips did this player put in (including ante)?"""
    contrib = 1  # ante
    parts   = history.split("/")
    act_num = 0
    for pi, part in enumerate(parts):
        bet_size = 2 if pi == 0 else 4
        for ch in part:
            if act_num % 2 == player:
                if ch in ("b", "r"):
                    contrib += bet_size
                elif ch == "c":
                    contrib += bet_size
            act_num += 1
    return contrib

def get_info_set_leduc(history, cards, player):
    """What player can observe: their card + (community card if revealed) + history."""
    parts = history.split("/")
    private = cards[player]
    comm    = cards[2] if len(parts) > 1 else -1
    return f"p{player}|{private}|{comm}|{history}"


class LeducCFR:
    """CFR+ solver for Leduc Poker."""
    from collections import defaultdict

    def __init__(self):
        from collections import defaultdict
        self.regrets      = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.t = 0

    def _regret_match(self, i_set, actions):
        r = self.regrets[i_set]
        pos = {a: max(r.get(a, 0.0), 0.0) for a in actions}
        total = sum(pos.values())
        if total > 0:
            return {a: pos[a] / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    def _cfr(self, cards, history, p0, p1):
        if is_terminal_leduc(history):
            return get_payoff_leduc(history, cards, 0)

        # Check if we need to deal community card (transition between rounds)
        parts = history.split("/")
        if len(parts) == 1 and _is_round_over(parts[0]) and "f" not in history:
            # Round 1 is over, move to round 2 — community card is already in cards
            return self._cfr(cards, history + "/", p0, p1)

        player  = current_player_leduc(history)
        actions = get_actions_leduc(history)
        if not actions:
            return get_payoff_leduc(history, cards, 0)

        i_set = get_info_set_leduc(history, cards, player)
        sigma = self._regret_match(i_set, actions)

        reach_i = p0 if player == 0 else p1
        for a in actions:
            self.strategy_sum[i_set][a] += self.t * reach_i * sigma[a]

        v = {}
        for a in actions:
            next_hist = history + a
            if player == 0:
                v[a] = self._cfr(cards, next_hist, p0 * sigma[a], p1)
            else:
                v[a] = self._cfr(cards, next_hist, p0, p1 * sigma[a])

        node_v   = sum(sigma[a] * v[a] for a in actions)
        cf_reach = p1 if player == 0 else p0

        for a in actions:
            delta = cf_reach * ((v[a] - node_v) if player == 0 else (-(v[a] - node_v)))
            self.regrets[i_set][a] = max(self.regrets[i_set][a] + delta, 0.0)

        return node_v

    def train(self, iterations):
        all_deals = get_all_deals()
        for _ in range(iterations):
            self.t += 1
            for cards in all_deals:
                self._cfr(list(cards), "", 1.0, 1.0)

    def get_full_strategy(self):
        result = {}
        for i_set in self.strategy_sum:
            actions = list(self.strategy_sum[i_set].keys())
            s = self.strategy_sum[i_set]
            total = sum(s.values())
            result[i_set] = {a: s[a] / total for a in actions} if total > 0 \
                            else {a: 1.0 / len(actions) for a in actions}
        return result