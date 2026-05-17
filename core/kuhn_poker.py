# core/kuhn_poker.py

# Cards: 1=Jack, 2=Queen, 3=King
CARDS = [1, 2, 3]
CARD_NAMES = {1: "J", 2: "Q", 3: "K"}

# Actions
CHECK = "c"
BET   = "b"
CALL  = "c"   # same symbol as check - context distinguishes them
FOLD  = "f"

# History strings tell us everything about a hand.
# "" = start
# "c" = player 1 checked
# "b" = player 1 bet
# "cb" = P1 check, P2 bet
# "bc" = P1 bet, P2 call
# "bf" = P1 bet, P2 fold
# "cc" = P1 check, P2 check -> showdown
# "cbc" = P1 check, P2 bet, P1 call -> showdown
# "cbf" = P1 check, P2 bet, P1 fold

def is_terminal(history):
    """Is this history a terminal (game-over) node?"""
    if history == "bc":   return True   # P1 bet, P2 called
    if history == "bf":   return True   # P1 bet, P2 folded
    if history == "cc":   return True   # both checked
    if history == "cbc":  return True   # check-bet-call
    if history == "cbf":  return True   # check-bet-fold
    return False

def get_actions(history):
    """What actions are available from this history?"""
    if is_terminal(history):
        return []
    if history == "":   return ["c", "b"]    # P1 first action
    if history == "c":  return ["c", "b"]    # P2 responds to check
    if history == "b":  return ["c", "f"]    # P2 responds to bet (c=call here)
    if history == "cb": return ["c", "f"]    # P1 responds to check-bet (c=call)
    return []

def current_player(history):
    """Whose turn is it? 0=Player1, 1=Player2"""
    if history in ["", "b", "cb"]:
        return 0 if history == "" or history == "cb" else 1
    if history == "c":
        return 1
    # More explicit:
    if history == "":   return 0
    if history == "c":  return 1
    if history == "b":  return 1
    if history == "cb": return 0
    return -1  # terminal

def get_payoff(history, cards, player):
    """
    Returns payoff for 'player' (0 or 1) at a terminal node.
    cards = [p1_card, p2_card]
    Pot starts at 2 (1 ante each). Bets are 1 chip.
    """
    c0, c1 = cards[0], cards[1]
    
    if history == "bc":
        # P1 bet, P2 called. Pot = 4. Showdown.
        winner = 0 if c0 > c1 else 1
        if player == winner:
            return 2   # win opponent's 2 chips
        else:
            return -2
    
    if history == "bf":
        # P1 bet, P2 folded. P1 wins pot.
        return 1 if player == 0 else -1
    
    if history == "cc":
        # Both checked. Pot = 2. Showdown.
        winner = 0 if c0 > c1 else 1
        return 1 if player == winner else -1
    
    if history == "cbc":
        # P1 check, P2 bet, P1 called. Pot = 4. Showdown.
        winner = 0 if c0 > c1 else 1
        return 2 if player == winner else -2
    
    if history == "cbf":
        # P1 check, P2 bet, P1 folded. P2 wins.
        return -1 if player == 0 else 1
    
    raise ValueError(f"Unknown terminal history: {history}")

def get_info_set(history, cards, player):
    """
    The information set key: what the current player can observe.
    = their own card + the history of actions.
    They CANNOT see opponent's card.
    """
    card = cards[player]
    return f"{CARD_NAMES[card]}:{history}"