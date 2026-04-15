# Example Datasets

## Datasets from Use Cases
The library ships with sample datasets that exemplify four unique recommendation scenarios.


| Dataset                 | Interactions      | Items          | Users         | Remarks
|-------------------------|----------------| ----------------| ----------------| ---------------
| **sample binary reward** | `sample_binary_reward_interactions`<br>Each row represents a user-item interaction with a binary reward (*OUTCOME*) indicating a successful conversion (1) or not (0), along with 7 item-level features. | `sample_binary_reward_items`<br>Unique item ids under *ITEM_ID* referenced in the interactions dataset. | `sample_binary_reward_users`<br>Each row is a unique user with 4 features, identified by *USER_ID*. | Can be used with **item-level** or **global reward** scorers.
| **sample continuous reward** | `sample_continuous_reward_interactions`<br>User-item interactions with a continuous reward (*OUTCOME*) representing engagement level (e.g., click count). | `sample_continuous_reward_items`<br>Unique items under *ITEM_ID* referenced in the interactions dataset. | `sample_continuous_reward_users`<br>Each row is a unique user with 8 features. | Used with **item-level** scorer.
| **sample multi-class scorer** | `sample_multi_class_interactions`<br>Each row shows whether a user selected a particular item. Since items are only assigned upon conversion, *OUTCOME* is always 1. | `sample_multi_class_items`<br>Each row is an item under *ITEM_ID* with 2 item-level features. | `sample_multi_class_users`<br>Each row is a unique user with 5 features. | *OUTCOME* is used for tracking only, not for model training.
| **sample multi-output scorer** | `sample_multi_output_interactions`<br>User interactions with all items are grouped together. Each row has a binary reward per user-item combination. | `sample_multi_output_items`<br>List of available items in column *ITEM_ID*. | `sample_multi_output_users`<br>Each row is a unique user with 6 features. | Only user features are used (no context or interaction features).

## Usage

Access sample datasets that match your use case:

```python
from skrec.examples.datasets import (
    sample_binary_reward_interactions,
    sample_binary_reward_items,
    sample_binary_reward_users,
)

interactions_df = sample_binary_reward_interactions.fetch_data()
users_df = sample_binary_reward_users.fetch_data()
items_df = sample_binary_reward_items.fetch_data()
```
