from enum import Enum


class StrategyFlag(str, Enum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"


class StrategyType(str, Enum):
    EPSILON_GREEDY = "epsilon_greedy"
    STATIC_ACTION = "static_action"
