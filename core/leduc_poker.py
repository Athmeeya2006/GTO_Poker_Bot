# core/leduc_poker.py
"""
Leduc Poker - complete implementation with game interface for generic exploitability.

Rules:
- 6 cards: two each of J(1), Q(2), K(3)
- 2 players, ante=1 each
- Round 1: bet size 2, max 1 raise per round
- Flop: community card revealed
- Round 2: bet size 4, max 1 raise per round
- Showdown: pair with community card beats no pair; higher card breaks ties

This is a significantly larger game than Kuhn (~936 info sets vs 12),
making it the real scaling test for CFR algorithms.
"""
from collections import defaultdict
from itertools import permutations

# Cards
J, Q, K = 1, 2, 3
DECK = [J, J, Q, Q, K, K]
CARD_NAMES = {1: "J", 2: "Q", 3: "K"}


def get_all_deals():
    """All possible (p0_card, p1_card, community_card) deals.

    With a 6-card deck and 3 distinct cards dealt, there are
    6 × 5 × 4 = 120 ordered deals.
    """
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


# ── Terminal / action / player logic ─────────────────────────────

def _is_round_over(round_hist):
    """Check if a single betting round is complete."""
    if not round_hist:
        return False
    if round_hist.endswith("f"):
        return True
    # All terminal round sequences
    terminal_rounds = {
        "cc", "bc", "rc",
        "cbc", "cbf",
        "brc", "brf",
        "crc", "crf",
        "cbrc", "cbrf",
        "brbc", "brbf",
    }
    return round_hist in terminal_rounds


def is_terminal_leduc(history):
    """
    History is a string. '/' separates rounds.
    Round actions: c=check/call, b=bet, f=fold, r=raise
    Terminal if: fold in any round, or both rounds complete.
    """
    if "f" in history:
        return True
    parts = history.split("/")
    if len(parts) < 2:
        # Only round 1 - check if round 1 is over (but no fold)
        # If round 1 is over without fold, we transition to round 2
        return False
    # Two rounds exist - check if round 2 is over
    r2 = parts[1]
    return _is_round_over(r2)


def get_actions_leduc(history):
    """Returns valid actions given current history."""
    if is_terminal_leduc(history):
        return []

    parts = history.split("/")
    r_hist = parts[-1]  # current round's history

    # Check if we need a round transition
    if len(parts) == 1 and _is_round_over(r_hist) and "f" not in history:
        return []  # need transition, handled by solver

    n_bets = r_hist.count("b") + r_hist.count("r")

    if r_hist == "":
        return ["c", "b"]       # first to act: check or bet
    if r_hist == "c":
        return ["c", "b"]       # second to act after check
    if r_hist.endswith("b") and not r_hist.endswith("rb"):
        if n_bets < 2:
            return ["c", "f", "r"]  # can raise
        return ["c", "f"]
    if r_hist.endswith("r"):
        return ["c", "f"]       # can only call or fold after raise
    if r_hist == "cb":
        return ["c", "f", "r"] if n_bets < 2 else ["c", "f"]

    return []


def current_player_leduc(history):
    """Determine whose turn it is. Alternates within each round, P0 first."""
    parts = history.split("/")
    r_hist = parts[-1]
    n_acts = len(r_hist)
    return n_acts % 2  # P0 acts first in each round


def hand_rank(private, community):
    """Higher = better hand. Pair beats high card."""
    if private == community:
        return 100 + private     # pair: higher pair wins
    return private               # no pair: just card rank


def get_payoff_leduc(history, cards, player):
    """
    cards = [p0_card, p1_card, community_card]
    Returns payoff for player (0 or 1).
    """
    p0_card, p1_card, comm = cards

    # Count contributions
    parts = history.split("/")
    r1 = parts[0]
    r2 = parts[1] if len(parts) > 1 else ""

    # Who wins
    if "f" in history:
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
            return 0  # tie: split pot

    player_contrib = _player_contribution(history, player)
    opponent_contrib = _player_contribution(history, 1 - player)

    if winner == player:
        return opponent_contrib
    else:
        return -player_contrib


def _find_folder(history):
    """Returns player index who folded."""
    parts = history.split("/")
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
    parts = history.split("/")
    act_num = 0
    pending_bet = 0  # track if there's an outstanding bet to call

    for pi, part in enumerate(parts):
        bet_size = 2 if pi == 0 else 4
        pending_bet = 0  # reset at round start

        for ch in part:
            acting_player = act_num % 2
            if ch == "b" or ch == "r":
                if acting_player == player:
                    # Pay any outstanding bet + add own bet
                    contrib += pending_bet + bet_size
                pending_bet = bet_size if acting_player != player else bet_size
                if acting_player != player:
                    pending_bet = bet_size
                else:
                    pending_bet = bet_size
            elif ch == "c":
                if acting_player == player and pending_bet > 0:
                    contrib += pending_bet
                pending_bet = 0
            act_num += 1

    return contrib


def get_info_set_leduc(history, cards, player):
    """What player can observe: their card + (community card if round 2) + history."""
    parts = history.split("/")
    private = cards[player]
    comm = cards[2] if len(parts) > 1 else -1
    return f"{CARD_NAMES.get(private, str(private))}|{CARD_NAMES.get(comm, str(comm))}|{history}"


# ── Game Interface for generic exploitability ─────────────────────

class LeducGameInterface:
    """Adapter for compute_exploitability_generic."""

    def __init__(self):
        self._all_deals = get_all_deals()

    def is_terminal(self, history):
        return is_terminal_leduc(history)

    def get_actions(self, history):
        # Handle round transition
        parts = history.split("/")
        if len(parts) == 1 and _is_round_over(parts[0]) and "f" not in history:
            return get_actions_leduc(history + "/")
        return get_actions_leduc(history)

    def current_player(self, history):
        return current_player_leduc(history)

    def get_payoff(self, history, cards, player):
        return get_payoff_leduc(history, cards, player)

    def get_info_set(self, history, cards, player):
        return get_info_set_leduc(history, cards, player)

    def get_all_deals(self):
        return self._all_deals


# ── CFR+ solver for Leduc ─────────────────────────────────────────

class LeducCFR:
    """CFR+ solver for Leduc Poker.

    Uses the same regret-clipping and linear averaging as the Kuhn CFR+ solver.
    Handles the two-round structure with community card revelation.
    """

    def __init__(self):
        self.regrets = defaultdict(lambda: defaultdict(float))
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

        # Handle round transition: round 1 over, move to round 2
        parts = history.split("/")
        if len(parts) == 1 and _is_round_over(parts[0]) and "f" not in history:
            return self._cfr(cards, history + "/", p0, p1)

        player = current_player_leduc(history)
        actions = get_actions_leduc(history)
        if not actions:
            return get_payoff_leduc(history, cards, 0)

        i_set = get_info_set_leduc(history, cards, player)
        sigma = self._regret_match(i_set, actions)

        # Linear weighting for strategy accumulation (CFR+)
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

        node_v = sum(sigma[a] * v[a] for a in actions)
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

    def get_average_strategy(self, info_set, actions):
        s = self.strategy_sum[info_set]
        total = sum(s.get(a, 0.0) for a in actions)
        if total > 0:
            return {a: s.get(a, 0.0) / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    def get_full_strategy(self):
        result = {}
        for i_set in self.strategy_sum:
            actions = list(self.strategy_sum[i_set].keys())
            s = self.strategy_sum[i_set]
            total = sum(s.values())
            result[i_set] = {a: s[a] / total for a in actions} if total > 0 \
                            else {a: 1.0 / len(actions) for a in actions}
        return result