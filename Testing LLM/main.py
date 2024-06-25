import os
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
import tiktoken
from  ADA_Similarity import EmbeddingSearch


def loading_df():

    df=pd.read_csv(os.path.join(os.getcwd(),'bill_sum_data.csv')) 
    print(df)

    #show a partial view of the csv, considering this colunm name
    df_bills = df[['text', 'summary', 'title']]
    print(df_bills)

    pd.options.mode.chained_assignment = None #https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#evaluation-order-matters

    # s is input text
    def normalize_text(s, sep_token = " \n "):
        s = re.sub(r'\s+',  ' ', s).strip()
        s = re.sub(r". ,","",s)
        # remove all instances of multiple spaces
        s = s.replace("..",".")
        s = s.replace(". .",".")
        s = s.replace("\n", "")
        s = s.strip()
        
        return s

    df_bills['text']= df_bills["text"].apply(lambda x : normalize_text(x))

    #showing number of tokens per colunm
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df_bills['n_tokens'] = df_bills["text"].apply(lambda x: len(tokenizer.encode(x)))
    df_bills = df_bills[df_bills.n_tokens<8192]
    len(df_bills)

    return df_bills


if __name__ == "__main__":


    df_bills = loading_df()
    searcher = EmbeddingSearch()
    res = searcher.search_docs(df_bills, "Can I get information on cable company tax revenue?", top_n=4)


