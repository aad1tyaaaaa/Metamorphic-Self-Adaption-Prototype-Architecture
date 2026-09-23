"""Stability Monitor and rollback policy (Phase 10).

Watches the recent configuration decisions, measures how often they switch, and
rolls back to the last stable configuration when switching becomes excessive.
"""

from collections import Counter, deque

from configs.configurations import CONFIG_NAMES, DEPTH_MAP

STABLE, WARNING, UNSTABLE = "STABLE", "WARNING", "UNSTABLE"


class StabilityMonitor:
    def __init__(self, window=6, warning_threshold=0.4, unstable_threshold=0.7,
                 enabled=True):
        """
        window              number of recent decisions considered
        warning_threshold   volatility above which the status is WARNING
        unstable_threshold  volatility above which a rollback is triggered
        enabled             False makes the monitor observe without acting,
                            which is the Phase 15 ablation (A7)
        """
        if not 0 <= warning_threshold <= unstable_threshold <= 1:
            raise ValueError("require 0 <= warning_threshold <= unstable_threshold <= 1")

        self.window = window
        self.warning_threshold = warning_threshold
        self.unstable_threshold = unstable_threshold
        self.enabled = enabled

        self.history = deque(maxlen=window)
        self.stable_configuration = None
        self.rollback_count = 0
        self.observations = 0
        self.switches = 0
        # Volatility is a rolling measure; keeping the whole trace means the
        # summary can report the run rather than whatever the last window held.
        self.volatility_history = []

    # ------------------------------------------------------------------

    def volatility(self):
        """Share of adjacent pairs in the window that changed configuration."""
        if len(self.history) < 2:
            return 0.0

        changes = sum(
            1 for previous, current in zip(self.history, list(self.history)[1:])
            if previous != current
        )
        return changes / (len(self.history) - 1)

    def status(self, volatility):
        if volatility > self.unstable_threshold:
            return UNSTABLE
        if volatility > self.warning_threshold:
            return WARNING
        return STABLE

    def observe(self, configuration):
        """Record a proposed configuration and return the configuration to use."""
        if configuration not in CONFIG_NAMES:
            raise ValueError(f"unknown configuration {configuration!r}")

        if self.history and self.history[-1] != configuration:
            self.switches += 1

        self.observations += 1
        self.history.append(configuration)

        volatility = self.volatility()
        status = self.status(volatility)

        rolled_back = False
        selected = configuration

        if status == UNSTABLE and self.enabled and self.stable_configuration is not None:
            # Compare with the previous stable configuration and revert to it.
            if self.stable_configuration != configuration:
                selected = self.stable_configuration
                self.rollback_count += 1
                rolled_back = True
                self.history[-1] = selected
                volatility = self.volatility()
                status = self.status(volatility)
        elif status == STABLE:
            # Remember the configuration the window has settled on.
            self.stable_configuration = Counter(self.history).most_common(1)[0][0]

        self.volatility_history.append(volatility)

        return {
            "proposed_configuration": configuration,
            "configuration": selected,
            "depth": DEPTH_MAP[selected],
            "status": status,
            "volatility": volatility,
            "rolled_back": rolled_back,
            "rollback_count": self.rollback_count,
            "stable_configuration": self.stable_configuration,
        }

    def summary(self):
        """Run-level summary.

        `volatility` is the mean over the run, not the final window: a run can
        end on a quiet window after switching heavily, and reporting only the
        last window would contradict the switch count.
        """
        trace = self.volatility_history or [0.0]
        mean_volatility = sum(trace) / len(trace)

        return {
            "observations": self.observations,
            "switches": self.switches,
            "switch_rate": self.switches / max(self.observations - 1, 1),
            "volatility": mean_volatility,
            "volatility_mean": mean_volatility,
            "volatility_max": max(trace),
            "volatility_final": self.volatility(),
            "rollback_count": self.rollback_count,
            "rollback_rate": self.rollback_count / max(self.observations, 1),
            "status": self.status(mean_volatility),
            "current_configuration": self.history[-1] if self.history else None,
        }

    def report(self):
        data = self.summary()
        return (
            f"Stability status: {data['status']}\n"
            f"Volatility: {data['volatility']:.2f} "
            f"(max {data['volatility_max']:.2f})\n"
            f"Rollback count: {data['rollback_count']}\n"
            f"Current configuration: {data['current_configuration']}"
        )


def demo():
    """Synthetic instability test: rollback must actually occur."""
    monitor = StabilityMonitor(window=6)

    # Settle on a stable configuration first.
    for _ in range(6):
        monitor.observe("medium")

    assert monitor.summary()["status"] == STABLE
    assert monitor.stable_configuration == "medium"
    assert monitor.rollback_count == 0

    # Then force rapid switching.
    events = [monitor.observe(config) for config in
              ["deep", "shallow", "deep", "shallow", "deep", "shallow"]]

    assert any(event["status"] in (WARNING, UNSTABLE) for event in events), "no instability detected"
    assert monitor.rollback_count > 0, "instability never triggered a rollback"
    assert all(event["configuration"] in CONFIG_NAMES for event in events)

    # A disabled monitor detects the same instability but never acts.
    passive = StabilityMonitor(window=6, enabled=False)
    for config in ["medium"] * 6 + ["deep", "shallow"] * 3:
        passive.observe(config)

    assert passive.rollback_count == 0
    assert passive.summary()["switches"] > 0

    print("stability_monitor demo OK")
    print(monitor.report())
    print(f"(passive monitor: {passive.summary()['switches']} switches, "
          f"{passive.rollback_count} rollbacks)")


if __name__ == "__main__":
    demo()
