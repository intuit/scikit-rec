from skrec.recommender.bandits.datatypes import StrategyType
from skrec.recommender.bandits.strategy.epsilon_greedy import EpsilonGreedy
from skrec.recommender.bandits.strategy.static_action import StaticAction


class StrategyFactory:
    @staticmethod
    def create(flavor: StrategyType, strategy_params):
        if flavor == StrategyType.EPSILON_GREEDY:
            return EpsilonGreedy(**strategy_params)
        elif flavor == StrategyType.STATIC_ACTION:
            return StaticAction(**strategy_params)
        else:
            raise KeyError(f"Cannot find strategy with type {flavor}")
