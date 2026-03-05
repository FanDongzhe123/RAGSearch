import os

import pandas as pd
import asyncio

from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.tokenizer.get_tokenizer import get_tokenizer

import time


INPUT_DIR = "/scratch/df2362/graphrag/graphrags/HotpotQA/output"
LANCEDB_URI = f"{INPUT_DIR}/lancedb"

COMMUNITY_REPORT_TABLE = "community_reports"
ENTITY_TABLE = "entities"
COMMUNITY_TABLE = "communities"
RELATIONSHIP_TABLE = "relationships"
COVARIATE_TABLE = "covariates"
TEXT_UNIT_TABLE = "text_units"
COMMUNITY_LEVEL = 2


# read nodes table to get community and degree data
entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
community_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_TABLE}.parquet")

entities = read_indexer_entities(entity_df, community_df, COMMUNITY_LEVEL)

# load description embeddings to an in-memory lancedb vectorstore
# to connect to a remote db, specify url and port values.
description_embedding_store = LanceDBVectorStore(
    vector_store_schema_config=VectorStoreSchemaConfig(
        index_name="default-entity-description"
    )
)
description_embedding_store.connect(db_uri=LANCEDB_URI)

print(f"Entity count: {len(entity_df)}")
entity_df.head()


relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
relationships = read_indexer_relationships(relationship_df)

print(f"Relationship count: {len(relationship_df)}")
relationship_df.head()



report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
reports = read_indexer_reports(report_df, community_df, COMMUNITY_LEVEL)

print(f"Report records: {len(report_df)}")
report_df.head()


text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
text_units = read_indexer_text_units(text_unit_df)

print(f"Text unit records: {len(text_unit_df)}")
text_unit_df.head()

os.environ["GRAPHRAG_API_KEY"] = "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"
api_key = os.environ["GRAPHRAG_API_KEY"]

chat_config = LanguageModelConfig(
    api_key=api_key,
    type=ModelType.Chat,
    model_provider="openai",
    model="gpt-4.1",
    max_retries=20,
)
chat_model = ModelManager().get_or_create_chat_model(
    name="local_search",
    model_type=ModelType.Chat,
    config=chat_config,
)

embedding_config = LanguageModelConfig(
    api_key=api_key,
    type=ModelType.Embedding,
    model_provider="openai",
    model="text-embedding-3-small",
    max_retries=20,
)

text_embedder = ModelManager().get_or_create_embedding_model(
    name="local_search_embedding",
    model_type=ModelType.Embedding,
    config=embedding_config,
)

tokenizer = get_tokenizer(chat_config)


context_builder = LocalSearchMixedContext(
    community_reports=reports,
    text_units=text_units,
    entities=entities,
    relationships=relationships,
    # if you did not run covariates during indexing, set this to None
    covariates=None,
    entity_text_embeddings=description_embedding_store,
    embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
    text_embedder=text_embedder,
    tokenizer=tokenizer,
)

local_context_params = {
    "text_unit_prop": 0.5,
    "community_prop": 0.1,
    "conversation_history_max_turns": 5,
    "conversation_history_user_turns_only": True,
    "top_k_mapped_entities": 5,
    "top_k_relationships": 5,
    "include_entity_rank": True,
    "include_relationship_weight": True,
    "include_community_rank": False,
    "return_candidate_context": False,
    "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
    "max_tokens": 4096,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
}

model_params = {
    "max_tokens": 2_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
    "temperature": 0.0,
}

start_time = time.time()
context = context_builder.build_context(
    query="Black Book starred the actress and writer of what heritage?",
    conversation_history=None,
    **local_context_params)
end_time = time.time()
print(f"Context Chunks:\n{context.context_chunks}")
print(len(context.context_chunks))
print(f"Time taken: {end_time - start_time} seconds")

# search_engine = LocalSearch(
#     model=chat_model,
#     context_builder=context_builder,
#     tokenizer=tokenizer,
#     model_params=model_params,
#     context_builder_params=local_context_params,
#     response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
# )

# async def main():
#     result = await search_engine.search(
#         "Black Book starred the actress and writer of what heritage?"
#     )
#     print(result)

# asyncio.run(main())