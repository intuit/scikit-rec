"""
Candidate retrieval for two-stage recommendation pipelines.

scikit-rec is a **ranking** library. The retriever layer is the first stage:
it narrows a large item catalog down to a manageable candidate set, which the
ranker then scores precisely.

-----------------------------------------------------------------------
Which retriever should I use?
-----------------------------------------------------------------------

Decision tree
~~~~~~~~~~~~~

Do you have item features (price, category, pre-encoded embeddings, etc.)?

  YES -> Are those features numeric or already encoded as numbers?
        YES -> ContentBasedRetriever
              Best for: cold-start users, new items added after training,
              catalogs that change frequently.
              Not for: purely ID-based catalogs with no item metadata.
        NO  -> Pre-encode your categorical features first, then
              ContentBasedRetriever.

  NO -> Do you have trained embeddings (MF, NCF, Two-Tower)?
       YES -> EmbeddingRetriever
             Best for: warm users, stable catalogs, personalized retrieval.
             Not for: new users or items unseen at training time.
       NO  -> PopularityRetriever
             Best for: getting started quickly, cold-start fallback,
             non-personalized use cases, any estimator including XGBoost.
             Not for: personalized recommendations (every user gets the
             same candidates).

Quick-reference table
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Retriever
     - Personalized?
     - Needs training?
     - New items?
     - New users?
     - XGBoost?
   * - EmbeddingRetriever
     - Yes
     - Yes
     - No
     - No
     - No
   * - ContentBasedRetriever
     - Partially *
     - No
     - Yes
     - Partially **
     - Yes
   * - PopularityRetriever
     - No
     - No
     - Yes
     - Yes
     - Yes
   * - External via item_subset
     - You decide
     - You decide
     - You decide
     - You decide
     - Yes

\\* Personalized via user interaction history — quality depends on how many
interactions exist per user.

\\** New users with at least one interaction get a profile vector; fully
cold users (zero interactions) fall back to the globally popular candidates.

When to combine retrievers
~~~~~~~~~~~~~~~~~~~~~~~~~~

A production system often needs multiple strategies:

- Warm user, known items   -> EmbeddingRetriever (personalized)
- New user or new item     -> ContentBasedRetriever (feature-aware fallback)
- No features, no model    -> PopularityRetriever (last resort)

A FallbackRetriever that chains multiple strategies in priority order is a
natural extension point (not shipped in v1.0).

When to skip built-in retrievers entirely
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``item_subset`` directly (pass no retriever to ``RankingRecommender``)
when:

- You already have a retrieval system (Elasticsearch, FAISS, BM25, rules).
- Your retrieval is driven by business rules (only in-stock items, etc.).
- You need retrieval logic that spans multiple data sources.

Example::

    candidates = elasticsearch.query(user_id, top_k=500)
    recommender.set_item_subset(candidates)
    recommendations = recommender.recommend(interactions=context_df, top_k=10)
    recommender.clear_item_subset()

The ``item_subset`` seam is always available regardless of which retriever
(if any) is attached to the recommender.
"""

from skrec.retriever.base_retriever import BaseCandidateRetriever
from skrec.retriever.content_based_retriever import ContentBasedRetriever
from skrec.retriever.embedding_retriever import EmbeddingRetriever
from skrec.retriever.popularity_retriever import PopularityRetriever

__all__ = [
    "BaseCandidateRetriever",
    "EmbeddingRetriever",
    "ContentBasedRetriever",
    "PopularityRetriever",
]
