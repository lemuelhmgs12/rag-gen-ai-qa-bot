"""
Builds the Chroma vector database from the help-desk FAQ.
Run from the project root:  uv run python -m src.ingest
"""
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src import config


def load_chunks():
    """Load the FAQ and split it into one chunk per Q&A pair."""
    loader = TextLoader(config.DATA_FILE, encoding="utf-8")
    docs = loader.load()
    text = docs[0].page_content

    splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=[("###", "id")],
                    strip_headers=True
                    )

    return splitter.split_text(text)
    


def build_vector_db(chunks):
    """Delete any old database, then embed the chunks and save them."""
    # start fresh on every run
    if config.VECTOR_DB_DIR.exists():
        shutil.rmtree(config.VECTOR_DB_DIR)

    #create the embeddings
    embeddings = OpenAIEmbeddings(model=config.EMEDDING_MODEL)

    # create vector database
    vectorstore = Chroma.from_documents(documents=chunks,
                                    embedding=embeddings,
                                    collection_metadata={"hnsw:space": "cosine"},
                                    persist_directory=config.VECTOR_DB_DIR,
                                    collection_name=config.COLLECTION_NAME)

    
    return vectorstore


def main():
    load_dotenv()

    chunks = load_chunks()
    print("Chunks to embed:", len(chunks))

    vectorstore = build_vector_db(chunks)
    print("Chunks in database:", len(vectorstore.get()["ids"]))


if __name__ == "__main__":
    main()