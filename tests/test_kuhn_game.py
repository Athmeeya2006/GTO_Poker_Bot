# tests/test_kuhn_game.py
import sys
sys.path.insert(0, '.')
from core.kuhn_poker import *

def test_terminal_detection():
    assert is_terminal("bc")  == True
    assert is_terminal("bf")  == True
    assert is_terminal("cc")  == True
    assert is_terminal("cbc") == True
    assert is_terminal("cbf") == True
    assert is_terminal("")    == False
    assert is_terminal("c")   == False
    assert is_terminal("b")   == False
    assert is_terminal("cb")  == False
    print("PASS: terminal detection")

def test_payoffs():
    # P1 has King (3), P2 has Jack (1)
    # P1 bet, P2 called -> P1 wins 2
    assert get_payoff("bc", [3,1], 0) == 2
    assert get_payoff("bc", [3,1], 1) == -2
    
    # P1 bet, P2 folded -> P1 wins 1
    assert get_payoff("bf", [1,3], 0) == 1   # even with losing card, wins by fold
    assert get_payoff("bf", [1,3], 1) == -1
    
    # P1 Jack, P2 Queen, both check -> P2 wins 1
    assert get_payoff("cc", [1,2], 0) == -1
    assert get_payoff("cc", [1,2], 1) == 1
    print("PASS: payoffs")

def test_info_sets():
    # P1 has King, history is empty
    key = get_info_set("", [3,2], 0)
    assert key == "K:", f"Got {key}"
    
    # P2 has Jack, P1 has bet
    key = get_info_set("b", [3,1], 1)
    assert key == "J:b", f"Got {key}"
    print("PASS: information sets")

def test_current_player():
    assert current_player("") == 0
    assert current_player("c") == 1
    assert current_player("b") == 1
    assert current_player("cb") == 0
    print("PASS: current player")

if __name__ == "__main__":
    test_terminal_detection()
    test_payoffs()
    test_info_sets()
    test_current_player()
    print("\nAll Kuhn game tests passed.")