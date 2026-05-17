# core/game_engine.py
"""
Interactive game engine - play against the GTO bot in real-time.

Wires together:
1. CFR solver (pre-computed Nash strategy)
2. Dirichlet opponent model (tracks YOUR patterns in real-time)
3. Strategy mixer (exploits your leaks when confident)
4. Range model (Bayesian hand inference)
5. SPRT leak detector (statistical significance gating)

This is the module that turns isolated components into an actual poker bot.
"""
import random
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.kuhn_poker import (
    CARDS, CARD_NAMES, is_terminal, get_actions,
    current_player, get_payoff, get_info_set
)
from core.cfr_plus import CFRPlus
from core.exploitability import compute_exploitability
from opponent_model.dirichlet_model import DirichletOpponentModel
from opponent_model.hypothesis_test import SPRTLeakDetector
from opponent_model.strategy_mixer import StrategyMixer
from opponent_model.range_model import RangeModel


# ── Action display mapping ─────────────────────────────────────

ACTION_NAMES = {
    "c": "check/call",
    "b": "bet",
    "f": "fold",
}


def _action_display(action, history):
    """Context-aware action name."""
    if action == "c":
        # 'c' after a bet means 'call', otherwise 'check'
        if "b" in history:
            return "call"
        return "check"
    return ACTION_NAMES.get(action, action)


class KuhnPokerBot:
    """
    A fully wired GTO poker bot for Kuhn Poker that:
    - Plays using pre-computed Nash strategy
    - Observes and models the opponent in real-time
    - Exploits detected leaks with statistical confidence gating
    - Tracks hand ranges via Bayesian inference

    This is NOT just a solver. This is a bot that plays.
    """

    def __init__(self, training_iters=50000, exploit_mode=True):
        """
        Args:
            training_iters: CFR+ iterations for Nash computation
            exploit_mode: if True, adapt to opponent leaks
        """
        self.exploit_mode = exploit_mode

        # 1. Compute Nash equilibrium
        print(f"Computing Nash equilibrium ({training_iters} CFR+ iterations)...")
        self.solver = CFRPlus()
        self.solver.train(training_iters)
        self.nash_strategy = self.solver.get_full_strategy()
        self.exploitability = compute_exploitability(self.nash_strategy)
        print(f"Nash computed. Exploitability: {self.exploitability:.6f}")

        # 2. Opponent modeling components
        self.opponent_model = DirichletOpponentModel(alpha_prior=1.0)
        self.leak_detector = SPRTLeakDetector(alpha=0.05, beta=0.05, epsilon=0.15)
        self.strategy_mixer = StrategyMixer(
            self.nash_strategy, confidence_threshold=0.6
        )

        # 3. Game state
        self.bot_player = -1  # set per hand
        self.hands_played = 0
        self.bot_bankroll = 0.0
        self.human_bankroll = 0.0

        # Per-position tracking: Kuhn Poker P0 has game value -1/18 ≈ -0.056,
        # so raw bankroll is misleading without controlling for position.
        self.human_hands_as_p0 = 0
        self.human_hands_as_p1 = 0
        self.human_bankroll_as_p0 = 0.0
        self.human_bankroll_as_p1 = 0.0

    def _bot_action(self, info_set, actions, history):
        """
        Choose bot's action using Nash + optional exploitation.

        In GTO mode: pure Nash strategy
        In exploit mode: blend Nash with counter-strategy based on
        detected opponent leaks
        """
        if not self.exploit_mode or self.hands_played < 10:
            # Pure GTO until we have enough data
            sigma = self.nash_strategy.get(
                info_set, {a: 1.0 / len(actions) for a in actions}
            )
        else:
            # Detect leak type
            leak_type = self._detect_leak_type()

            sigma, confidence = self.strategy_mixer.get_mixed_strategy(
                info_set, actions, self.opponent_model,
                self.nash_strategy, leak_type=leak_type
            )

        # Sample action from strategy
        r = random.random()
        cumulative = 0.0
        for a in actions:
            cumulative += sigma.get(a, 0.0)
            if r < cumulative:
                return a
        return actions[-1]

    def _detect_leak_type(self):
        """Analyze opponent model to determine leak type."""
        leaks = self.leak_detector.get_detected_leaks()
        if not leaks:
            return None

        # Check for over-folding
        fold_leaks = [l for l in leaks if l[1] == "f"]
        if fold_leaks:
            return "folds_too_much"

        # Check for over-calling
        call_leaks = [l for l in leaks if l[1] == "c"]
        if call_leaks:
            return "calls_too_much"

        return None

    def _observe_opponent_action(self, info_set, action, actions):
        """Update all opponent modeling components after observing an action."""
        self.opponent_model.observe(info_set, action)

        # SPRT update for each action at this info set
        for a in actions:
            nash_freq = self.nash_strategy.get(info_set, {}).get(a, 1.0 / len(actions))
            self.leak_detector.update(info_set, a, action, nash_freq)

    def play_hand(self, verbose=True):
        """
        Play one hand of Kuhn Poker.
        Returns: (human_payoff, bot_payoff)
        """
        # Deal cards
        cards = random.sample(CARDS, 2)

        # Randomly assign positions
        self.bot_player = random.randint(0, 1)
        human_player = 1 - self.bot_player

        human_card = cards[human_player]
        bot_card = cards[self.bot_player]

        if verbose:
            print(f"\n{'─' * 40}")
            pos_name = "Player 1 (acts first)" if human_player == 0 else "Player 2"
            bot_pos = "Player 1 (acts first)" if self.bot_player == 0 else "Player 2"
            print(f"You are {pos_name} | Bot is {bot_pos}")
            print(f"Your card: {CARD_NAMES[human_card]}")
            print(f"Pot: 2 chips (1 ante each)")

        history = ""

        while not is_terminal(history):
            player = current_player(history)
            actions = get_actions(history)
            info_set = get_info_set(history, cards, player)

            if player == human_player:
                # Human's turn
                if verbose:
                    print(f"\nYour turn. Actions: ", end="")
                    action_strs = [
                        f"[{a}] {_action_display(a, history)}" for a in actions
                    ]
                    print(" | ".join(action_strs))

                while True:
                    choice = input("Your action: ").strip().lower()
                    if choice in actions:
                        break
                    print(f"Invalid. Choose from: {actions}")

                history += choice
                if verbose:
                    print(f"You {_action_display(choice, history)}.")

                # Observe human action for modeling
                opponent_info_set = get_info_set(
                    history[:-1], cards, human_player
                )
                self._observe_opponent_action(
                    opponent_info_set, choice, actions
                )

            else:
                # Bot's turn
                bot_info_set = get_info_set(history, cards, self.bot_player)
                action = self._bot_action(bot_info_set, actions, history)
                history += action
                if verbose:
                    print(f"Bot {_action_display(action, history)}s.")

        # Hand is over
        human_payoff = get_payoff(history, cards, human_player)
        bot_payoff = get_payoff(history, cards, self.bot_player)

        self.human_bankroll += human_payoff
        self.bot_bankroll += bot_payoff
        self.hands_played += 1

        # Track per-position results
        if human_player == 0:
            self.human_hands_as_p0 += 1
            self.human_bankroll_as_p0 += human_payoff
        else:
            self.human_hands_as_p1 += 1
            self.human_bankroll_as_p1 += human_payoff

        if verbose:
            print(f"\n--- Result ---")
            print(f"Bot's card: {CARD_NAMES[bot_card]}")
            result_str = "You win!" if human_payoff > 0 else (
                "Bot wins!" if human_payoff < 0 else "Push!"
            )
            print(f"{result_str} ({human_payoff:+d} chips)")
            print(f"Running score: You {self.human_bankroll:+.0f} | "
                  f"Bot {self.bot_bankroll:+.0f} | "
                  f"Hands: {self.hands_played}")

        return human_payoff, bot_payoff

    def show_opponent_model(self):
        """Display current opponent model state."""
        print(f"\n{'═' * 50}")
        print("  OPPONENT MODEL STATUS")
        print(f"{'═' * 50}")
        print(f"Hands observed: {self.hands_played}")

        self.opponent_model.summary(self.nash_strategy)

        leaks = self.leak_detector.get_detected_leaks()
        if leaks:
            print(f"\nDetected leaks (SPRT confirmed):")
            for i_set, action, n_obs in leaks:
                print(f"  {i_set} | action={action} | {n_obs} observations")
        else:
            print(f"\nNo statistically confirmed leaks yet.")

        total_kl = self.opponent_model.total_kl_divergence(self.nash_strategy)
        print(f"\nTotal KL divergence from Nash: {total_kl:.4f}")

        if self.exploit_mode:
            leak_type = self._detect_leak_type()
            if leak_type:
                print(f"Current exploitation target: {leak_type}")
            else:
                print("Playing GTO (no exploitable leak confirmed)")

    def run_session(self, max_hands=None):
        """
        Run an interactive poker session.

        Args:
            max_hands: stop after this many hands (None = play until quit)
        """
        print(f"\n{'═' * 50}")
        print("  KUHN POKER - Human vs GTO Bot")
        print(f"{'═' * 50}")
        print(f"Bot exploitability: {self.exploitability:.6f} (0 = perfect Nash)")
        if self.exploit_mode:
            print("Exploit mode: ON - bot will adapt to your patterns")
        else:
            print("Exploit mode: OFF - bot plays pure GTO")
        print(f"\nCommands during play:")
        print(f"  'q' or 'quit' - end session")
        print(f"  'm' or 'model' - show opponent model")
        print(f"  's' or 'stats' - show session stats")
        print(f"\nLet's play!\n")

        try:
            while max_hands is None or self.hands_played < max_hands:
                try:
                    self.play_hand(verbose=True)
                except EOFError:
                    break

                # Post-hand menu
                cmd = input("\n[Enter] next hand | [m]odel | [s]tats | [q]uit: ").strip().lower()
                if cmd in ("q", "quit"):
                    break
                elif cmd in ("m", "model"):
                    self.show_opponent_model()
                elif cmd in ("s", "stats"):
                    self._show_stats()

        except KeyboardInterrupt:
            pass

        self._show_final_results()

    def _show_stats(self):
        """Show session statistics with position-adjusted win rates."""
        print(f"\n--- Session Stats ---")
        print(f"Hands played: {self.hands_played}")
        print(f"Your bankroll: {self.human_bankroll:+.0f}")
        print(f"Bot bankroll:  {self.bot_bankroll:+.0f}")
        if self.hands_played > 0:
            print(f"Your avg chips/hand: {self.human_bankroll / self.hands_played:+.3f}")
            print(f"Bot avg chips/hand:  {self.bot_bankroll / self.hands_played:+.3f}")

        # Position-adjusted stats
        # Kuhn game value is -1/18 from P0 perspective. Controlling for
        # position removes the systematic disadvantage from first-mover.
        KUHN_GAME_VALUE = -1.0 / 18.0  # P0's expected value at Nash
        print(f"\n--- Position-Adjusted Stats ---")
        if self.human_hands_as_p0 > 0:
            raw_p0 = self.human_bankroll_as_p0 / self.human_hands_as_p0
            adj_p0 = raw_p0 - KUHN_GAME_VALUE  # remove P0 disadvantage
            print(f"As P0 (first):  {self.human_hands_as_p0} hands, "
                  f"raw={raw_p0:+.3f}, adjusted={adj_p0:+.3f} chips/hand")
        if self.human_hands_as_p1 > 0:
            raw_p1 = self.human_bankroll_as_p1 / self.human_hands_as_p1
            adj_p1 = raw_p1 + KUHN_GAME_VALUE  # P1 has +1/18 advantage
            print(f"As P1 (second): {self.human_hands_as_p1} hands, "
                  f"raw={raw_p1:+.3f}, adjusted={adj_p1:+.3f} chips/hand")

    def _show_final_results(self):
        """Show end-of-session summary with position-adjusted stats."""
        print(f"\n{'═' * 50}")
        print("  SESSION COMPLETE")
        print(f"{'═' * 50}")
        print(f"Hands played:  {self.hands_played}")
        print(f"Your bankroll: {self.human_bankroll:+.0f}")
        print(f"Bot bankroll:  {self.bot_bankroll:+.0f}")
        if self.hands_played > 0:
            print(f"Your avg:      {self.human_bankroll / self.hands_played:+.3f} chips/hand")

            # Position breakdown
            KUHN_GAME_VALUE = -1.0 / 18.0
            print(f"\n--- Position Breakdown ---")
            print(f"  P0 (first to act) has EV = {KUHN_GAME_VALUE:+.4f} at Nash")
            if self.human_hands_as_p0 > 0:
                raw = self.human_bankroll_as_p0 / self.human_hands_as_p0
                print(f"  You as P0: {self.human_hands_as_p0} hands, {raw:+.3f} chips/hand")
            if self.human_hands_as_p1 > 0:
                raw = self.human_bankroll_as_p1 / self.human_hands_as_p1
                print(f"  You as P1: {self.human_hands_as_p1} hands, {raw:+.3f} chips/hand")

        self.show_opponent_model()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Play Kuhn Poker against a GTO bot with adaptive exploitation"
    )
    parser.add_argument(
        "--hands", type=int, default=None,
        help="Maximum number of hands to play (default: unlimited)"
    )
    parser.add_argument(
        "--iters", type=int, default=50000,
        help="CFR+ training iterations (default: 50000)"
    )
    parser.add_argument(
        "--no-exploit", action="store_true",
        help="Disable exploitation (pure GTO mode)"
    )
    args = parser.parse_args()

    bot = KuhnPokerBot(
        training_iters=args.iters,
        exploit_mode=not args.no_exploit
    )
    bot.run_session(max_hands=args.hands)


if __name__ == "__main__":
    main()
