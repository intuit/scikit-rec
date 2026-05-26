ITEM_ID_NAME = "ITEM_ID"
USER_ID_NAME = "USER_ID"
LABEL_NAME = "OUTCOME"
ITEM_PREFIX = "ITEM_"
DEBUG_COLUMNS = [ITEM_ID_NAME, USER_ID_NAME]
OUTCOME_PREFIX = LABEL_NAME + "_"
# Observed-label prefix at inference for real-time-label conditioning
# (v3 conditional MixedTypeMultiTargetScorer estimators). Mapping:
#   ITEM_<suffix> (target) ↔ OBSERVED_<suffix> (observed input)
# In v2, OBSERVED_* columns are rejected at inference by the scorer's
# validator. In v3 they are honored by ConditionalMultiTargetEstimator
# implementations.
OBSERVED_PREFIX = "OBSERVED_"
USER_EMBEDDING_NAME = "EMBEDDING"
ITEM_EMBEDDING_NAME = "ITEM_EMBEDDING"
TIMESTAMP_COL = "TIMESTAMP"
