"""JSON battle recording and headless ASCII replay."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from knight_boss.game.engine import GameEngine
from knight_boss.game.types import Action


class BattleReplayRecorder:
    def __init__(
        self,
        engine: GameEngine,
        *,
        seed: int,
        policies: dict[str, str],
    ) -> None:
        self.replay: dict[str, Any] = {
            "schema_version": 1,
            "seed": seed,
            "alpha_h": engine.alpha_h,
            "alpha_d": engine.alpha_d,
            "game_config": asdict(engine.config),
            "policies": dict(policies),
            "initial_state": capture_state(engine),
            "frames": [],
        }

    def record_step(
        self,
        engine: GameEngine,
        actions: dict[str, int | Action],
        rewards: dict[str, float],
        infos: dict[str, dict[str, Any]],
        terminations: dict[str, bool],
        truncations: dict[str, bool],
    ) -> None:
        self.replay["frames"].append(
            {
                "step": engine.step_count,
                "actions": {
                    agent: Action(actions[agent]).name for agent in ("knight", "boss")
                },
                "state": capture_state(engine),
                "events": list(infos["knight"]["events"]),
                "rewards": dict(rewards),
                "terminations": dict(terminations),
                "truncations": dict(truncations),
            }
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as replay_file:
            json.dump(self.replay, replay_file, indent=2)
        return destination


def capture_state(engine: GameEngine) -> dict[str, Any]:
    fighter_states = {}
    for agent, fighter in engine.fighters.items():
        fighter_states[agent] = {
            "position": list(fighter.position),
            "health": fighter.health,
            "max_health": fighter.max_health,
            "attacks": {
                "basic": {
                    "damage": fighter.basic_damage,
                    "range": fighter.basic_range,
                    "cooldown": fighter.basic_cooldown,
                    "cooldown_duration": fighter.basic_cooldown_duration,
                },
                "heavy": {
                    "damage": fighter.heavy_damage,
                    "range": fighter.heavy_range,
                    "cooldown": fighter.heavy_cooldown,
                    "cooldown_duration": fighter.heavy_cooldown_duration,
                },
            },
        }
    return {
        "step": engine.step_count,
        "fighters": fighter_states,
        "windups": {
            agent: {
                "target": attack.target,
                "direction": list(attack.direction),
                "remaining_steps": attack.remaining_steps,
            }
            for agent, attack in engine.pending_heavy.items()
        },
        "outcome": engine.outcome.value if engine.outcome is not None else None,
        "terminated": engine.terminated,
        "truncated": engine.truncated,
    }


def load_replay(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(encoding="utf-8") as replay_file:
        replay = json.load(replay_file)
    if replay.get("schema_version") != 1:
        raise ValueError("Unsupported battle replay schema version.")
    if not isinstance(replay.get("frames"), list) or "initial_state" not in replay:
        raise ValueError("Battle replay is missing required state or frames.")
    return replay


def render_replay_state(
    replay: dict[str, Any],
    state: dict[str, Any],
    *,
    actions: dict[str, str] | None = None,
    events: list[str] | None = None,
) -> str:
    config = replay["game_config"]
    rows = int(config["rows"])
    columns = int(config["columns"])
    board = [["." for _ in range(columns)] for _ in range(rows)]
    fighters = state["fighters"]
    board[tuple(fighters["knight"]["position"])[0]][
        tuple(fighters["knight"]["position"])[1]
    ] = "K"
    board[tuple(fighters["boss"]["position"])[0]][
        tuple(fighters["boss"]["position"])[1]
    ] = "B"
    lines = [" ".join(row) for row in board]
    lines.append(
        _fighter_summary("Knight", fighters["knight"])
        + " | "
        + _fighter_summary("Boss", fighters["boss"])
    )
    if actions:
        lines.append(
            f"Actions: Knight={actions.get('knight', 'N/A')}, "
            f"Boss={actions.get('boss', 'N/A')}"
        )
    windups = state.get("windups", {})
    lines.append(
        "Wind-ups: "
        + (
            ", ".join(
                f"{agent} -> {pending['target']} dir={tuple(pending['direction'])} "
                f"remaining={pending['remaining_steps']}"
                for agent, pending in windups.items()
            )
            if windups
            else "none"
        )
    )
    if events:
        lines.append(f"Events: {', '.join(events)}")
    lines.append(
        f"Step: {state['step']}/{config['max_steps']} | "
        f"Alpha_H={replay['alpha_h']} | Alpha_D={replay['alpha_d']}"
    )
    if state.get("outcome") is not None:
        lines.append(f"Outcome: {state['outcome']}")
    return "\n".join(lines)


def play_replay(path: str | Path, *, delay_seconds: float = 0.0) -> None:
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be non-negative.")
    replay = load_replay(path)
    print(render_replay_state(replay, replay["initial_state"]))
    for frame in replay["frames"]:
        if delay_seconds:
            time.sleep(delay_seconds)
        print()
        print(
            render_replay_state(
                replay,
                frame["state"],
                actions=frame["actions"],
                events=frame["events"],
            )
        )


def _fighter_summary(name: str, fighter: dict[str, Any]) -> str:
    basic = fighter["attacks"]["basic"]
    heavy = fighter["attacks"]["heavy"]
    health_fraction = fighter["health"] / fighter["max_health"]
    filled = round(health_fraction * 10)
    health_bar = "#" * filled + "-" * (10 - filled)
    return (
        f"{name} HP=[{health_bar}] {fighter['health']}/{fighter['max_health']} "
        f"pos={tuple(fighter['position'])} "
        f"basic={basic['damage']} dmg/range {basic['range']}/"
        f"CD {basic['cooldown']}/{basic['cooldown_duration']} "
        f"heavy={heavy['damage']} dmg/range {heavy['range']}/"
        f"CD {heavy['cooldown']}/{heavy['cooldown_duration']}"
    )
