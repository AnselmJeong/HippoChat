from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utilities import ChromaDB
import os
import argparse


embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, 
                                               chunk_overlap=48)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a directory containing pdf files",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--directory", default="./", 
                        help="directory to search for pdf files")
    parser.add_argument("-t", "--target", default="./.db", 
                        help="directory for storing DB")
    
    
    args = parser.parse_args()
    
    directory = args.directory
    target = args.target
    
    db = ChromaDB(target, prefix=None, embeddings=embeddings, text_splitter=text_splitter)
    try:
        db.ingest_directory(directory)
        print(f"Ingestion succeeded. DB is stored at {target}")
    except Exception as e:
        print("Failed to ingest pdf files. Error: ", e)
    
    