import os

from skrec.dataset.interactions_dataset import (
    InteractionMultiClassDataset,
    InteractionMultiOutputDataset,
    InteractionsDataset,
)
from skrec.dataset.items_dataset import ItemsDataset
from skrec.dataset.users_dataset import UsersDataset

rt = os.path.dirname(__file__)

# load sample dataset with binary reward
binary_reward_data_path = os.path.join(rt, "sample_binary_reward")
sample_binary_reward_interactions = InteractionsDataset(
    data_location=os.path.join(binary_reward_data_path, "interactions.csv"),
    client_schema_path=os.path.join(binary_reward_data_path, "interactions_schema.yaml"),
)
sample_binary_reward_items = ItemsDataset(
    data_location=os.path.join(binary_reward_data_path, "items.csv"),
    client_schema_path=os.path.join(binary_reward_data_path, "items_schema.yaml"),
)
sample_binary_reward_users = UsersDataset(
    data_location=os.path.join(binary_reward_data_path, "users.csv"),
    client_schema_path=os.path.join(binary_reward_data_path, "users_schema.yaml"),
)

# ----------------------------------------------------------------------------------------------------------------------

# load sample dataset with continuous reward
continuous_reward_data_path = os.path.join(rt, "sample_continuous_reward")
sample_continuous_reward_interactions = InteractionsDataset(
    data_location=os.path.join(continuous_reward_data_path, "interactions.csv"),
    client_schema_path=os.path.join(continuous_reward_data_path, "interactions_schema.yaml"),
)
sample_continuous_reward_items = ItemsDataset(
    data_location=os.path.join(continuous_reward_data_path, "items.csv"),
    client_schema_path=os.path.join(continuous_reward_data_path, "items_schema.yaml"),
)
sample_continuous_reward_users = UsersDataset(
    data_location=os.path.join(continuous_reward_data_path, "users.csv"),
    client_schema_path=os.path.join(continuous_reward_data_path, "users_schema.yaml"),
)

# ----------------------------------------------------------------------------------------------------------------------

# load multi-output sample dataset
multi_output_data_path = os.path.join(rt, "sample_multi_output")
sample_multi_output_interactions = InteractionMultiOutputDataset(
    data_location=os.path.join(multi_output_data_path, "interactions.csv"),
    client_schema_path=os.path.join(multi_output_data_path, "interactions_schema.yaml"),
)

# ----------------------------------------------------------------------------------------------------------------------

# load multi-output multi-class sample dataset
multi_output_multi_class_data_path = os.path.join(rt, "sample_multi_output_multi_class")
sample_multi_output_multi_class_interactions = InteractionMultiOutputDataset(
    data_location=os.path.join(multi_output_multi_class_data_path, "interactions.csv"),
    client_schema_path=os.path.join(multi_output_multi_class_data_path, "interactions_schema.yaml"),
)

# ----------------------------------------------------------------------------------------------------------------------

# load multi-class sample dataset
multi_class_data_path = os.path.join(rt, "sample_multi_class")
sample_multi_class_interactions = InteractionMultiClassDataset(
    data_location=os.path.join(multi_class_data_path, "interactions_multi_class.csv"),
    client_schema_path=os.path.join(multi_class_data_path, "interactions_schema.yaml"),
)

# ----------------------------------------------------------------------------------------------------------------------

# load multi-output sample dataset
multi_outcome_data_path = os.path.join(rt, "sample_multi_outcome")
sample_multi_outcome_interactions = InteractionsDataset(
    data_location=os.path.join(multi_outcome_data_path, "interactions_multi_outcome.csv")
)

# ----------------------------------------------------------------------------------------------------------------------

# load sample uplift dataset
uplift_data_path = os.path.join(rt, "sample_uplift")
sample_uplift_interactions = InteractionsDataset(
    data_location=os.path.join(uplift_data_path, "interactions.csv"),
    client_schema_path=os.path.join(uplift_data_path, "interactions_schema.yaml"),
)
sample_uplift_users = UsersDataset(
    data_location=os.path.join(uplift_data_path, "users.csv"),
    client_schema_path=os.path.join(uplift_data_path, "users_schema.yaml"),
)
sample_uplift_interactions_with_feats = InteractionsDataset(
    data_location=os.path.join(uplift_data_path, "interactions_with_feats.csv"),
    client_schema_path=os.path.join(uplift_data_path, "interactions_schema_with_feats.yaml"),
)

# Export all sample datasets
__all__ = [
    # Binary reward datasets
    "sample_binary_reward_interactions",
    "sample_binary_reward_items",
    "sample_binary_reward_users",
    # Continuous reward datasets
    "sample_continuous_reward_interactions",
    "sample_continuous_reward_items",
    "sample_continuous_reward_users",
    # Multi-output datasets
    "sample_multi_output_interactions",
    "sample_multi_output_multi_class_interactions",
    # Multi-class datasets
    "sample_multi_class_interactions",
    # Multi-outcome datasets
    "sample_multi_outcome_interactions",
    # Uplift datasets
    "sample_uplift_interactions",
    "sample_uplift_users",
    "sample_uplift_interactions_with_feats",
]
