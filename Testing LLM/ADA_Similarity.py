import os
import re
import requests
import sys
from num2words import num2words
import pandas as pd
import numpy as np
import tiktoken
from openai import AzureOpenAI

class EmbeddingSearch:
    def __init__(self):
        self.client = self.model_configuration()

    def model_configuration(self):
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        return client

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_embedding(self, text, model="text-embedding-ada-002"):
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding

    def search_docs(self, df, user_query, top_n=4, to_print=True):
        embedding = self.get_embedding(user_query, model="text-embedding-ada-002")
        df["similarities"] = df.ada_v2.apply(lambda x: self.cosine_similarity(x, embedding))

        res = df.sort_values("similarities", ascending=False).head(top_n)
        
        if to_print:
            print(res)
            res.to_csv('search_results.csv', index=False)
        
        return res

