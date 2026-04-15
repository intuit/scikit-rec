from skrec.evaluator.datatypes import RecommenderEvaluatorType


class EvaluatorCategories:
    """Simple class to categorize evaluators based on probability requirements"""

    evaluators_with_probabilities = {
        RecommenderEvaluatorType.IPS,
        RecommenderEvaluatorType.SNIPS,
        RecommenderEvaluatorType.DR,
        RecommenderEvaluatorType.POLICY_WEIGHTED,
        RecommenderEvaluatorType.DIRECT_METHOD,
    }

    evaluators_without_probabilities = {RecommenderEvaluatorType.SIMPLE, RecommenderEvaluatorType.REPLAY_MATCH}

    @classmethod
    def requires_probability(cls, evaluator_type: RecommenderEvaluatorType) -> bool:
        """Check if an evaluator requires probability information"""
        return evaluator_type in cls.evaluators_with_probabilities
