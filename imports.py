from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice
import wikipedia
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
import os 
from haystack import Pipeline
import numpy as np 
from openai import OpenAI 
from collections import Counter 
from matplotlib import pyplot as plt 
from haystack.components.routers import FileTypeRouter
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from typing import Any, Dict, List, Optional, Union
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.rankers import TransformersSimilarityRanker
import time 
from openAI_API_key import *
import pdb
import wfdb 
# import 

EMBEDDER_MODEL = "thenlper/gte-large"
EMBEDDER_MODEL_LOCAL = "/home/ant/.cache/huggingface/hub/models--thenlper--gte-large/snapshots/58578616559541da766b9b993734f63bcfcfc057"
RANKER_MODEL = "BAAI/bge-reranker-base"
RANKER_MODEL_LOCAL = "/home/ant/.cache/huggingface/hub/models--BAAI--bge-reranker-base/snapshots/580465186bcc87f862a9b2f9003d720af2377980"
MODEL = {"gpt3.5": "gpt-3.5-turbo", "gpt4": "gpt-4-turbo-preview"}


# id2labels = {
#     1: "WALKING",
#     2: "WALKING_UPSTAIRS",
#     3: "WALKING_DOWNSTAIRS",
#     4: "SITTING",
#     5: "STANDING",
#     6: "LAYING"
# }
id2labels = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
    7: "STAND_TO_SIT",
    8: "SIT_TO_STAND",
    9: "SIT_TO_LIE",
    10: "LIE_TO_SIT",
    11: "STAND_TO_LIE",
    12: "LIE_TO_STAND"}

label2ids = {
    "WALKING": 1,
    "WALKING_UPSTAIRS": 2,
    "WALKING_DOWNSTAIRS": 3,
    "SITTING": 4,
    "STANDING": 5,
    "LAYING": 6,
    "STAND_TO_SIT": 7,
    "SIT_TO_STAND": 8,
    "SIT_TO_LIE": 9,
    "LIE_TO_SIT": 10,
    "STAND_TO_LIE": 11,
    "LIE_TO_STAND": 12
}
# label2ids = {
#     "WALKING": 1,
#     "WALKING_UPSTAIRS": 2,
#     "WALKING_DOWNSTAIRS": 3,
#     "SITTING": 4,
#     "STANDING": 5,
#     "LAYING": 6
# }


