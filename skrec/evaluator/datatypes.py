from enum import Enum


class RecommenderEvaluatorType(str, Enum):
    SIMPLE = "simple"
    REPLAY_MATCH = "replay_match"
    IPS = "IPS"
    DR = "DR"
    DIRECT_METHOD = "direct_method"
    SNIPS = "SNIPS"
    POLICY_WEIGHTED = "policy_weighted"
