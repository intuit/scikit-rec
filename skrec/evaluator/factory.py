from skrec.evaluator.base_evaluator import BaseRecommenderEvaluator
from skrec.evaluator.datatypes import RecommenderEvaluatorType
from skrec.evaluator.direct_method import DirectMethodEvaluator
from skrec.evaluator.doubly_robust import DREvaluator
from skrec.evaluator.inverse_propensity_score import IPSEvaluator
from skrec.evaluator.policy_weighted import PolicyWeightedEvaluator
from skrec.evaluator.replay_match import ReplayMatchEvaluator
from skrec.evaluator.simple import SimpleEvaluator
from skrec.evaluator.snips import SNIPSEvaluator


class RecommenderEvaluatorFactory:
    @staticmethod
    def create(flavor: RecommenderEvaluatorType, *args, **kwargs) -> BaseRecommenderEvaluator:
        if flavor == RecommenderEvaluatorType.SIMPLE:
            return SimpleEvaluator(*args, **kwargs)
        elif flavor == RecommenderEvaluatorType.REPLAY_MATCH:
            return ReplayMatchEvaluator(*args, **kwargs)
        elif flavor == RecommenderEvaluatorType.IPS:
            return IPSEvaluator(*args, **kwargs)
        elif flavor == RecommenderEvaluatorType.DR:
            return DREvaluator(*args, **kwargs)
        elif flavor == RecommenderEvaluatorType.DIRECT_METHOD:
            return DirectMethodEvaluator(*args, **kwargs)
        elif flavor == RecommenderEvaluatorType.SNIPS:
            return SNIPSEvaluator(*args, **kwargs)
        elif flavor == RecommenderEvaluatorType.POLICY_WEIGHTED:
            return PolicyWeightedEvaluator(*args, **kwargs)
        else:
            raise KeyError(f"Cannot find evaluator with type {flavor}")
