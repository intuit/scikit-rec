from typing import TYPE_CHECKING

from numpy.typing import NDArray
from pandas import DataFrame

from skrec.estimator.base_estimator import BaseEstimator

if TYPE_CHECKING:
    _Base = BaseEstimator
else:
    _Base = object


class NumpyPredictorMixin(_Base):
    def _process_for_predict(self, X: DataFrame) -> NDArray:
        return super()._process_for_predict(X).values
