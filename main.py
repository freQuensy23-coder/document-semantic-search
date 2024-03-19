from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import BM25Retriever
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.vectorstores import FAISS
import gradio as gr
import re

model = "msmarco-distilbert-base-tas-b"
embeddings = SentenceTransformerEmbeddings(model_name=model)
prev_files = None
retriever = None


def handle_files_and_query(query, files, chunk_overlap=50, token_per_chunk=256, bm_25_answers=200):
    results = ""
    global prev_files, retriever
    files = [f.name for f in files]
    if files is not None and files != prev_files:
        documents = []
        prev_files = files
        for file in files:
            documents.extend(
                PyMuPDFLoader(file).
                load_and_split(SentenceTransformersTokenTextSplitter(model_name=model,
                                                                     chunk_overlap=chunk_overlap,
                                                                     tokens_per_chunk=token_per_chunk)))
        retriever = BM25Retriever.from_documents(documents, k=bm_25_answers)
        results += "Index created successfully!\n"
        print("Index created successfully!")
    elif files is None:
        print("No files uploaded.")
    else:
        print("Reusing index since no files changed.")

    print(f"Query: {query}")
    if query:
        search_results = retriever.get_relevant_documents(query)
        pattern = r'[^\\/]+$'  # pattern to get filename from filepath
        reranked_results = FAISS.from_documents(search_results, embeddings,
                                                distance_strategy=DistanceStrategy.COSINE).similarity_search(query,
                                                                                                             k=25)
        results = "\n".join([
            f"Source: {re.search(pattern, result.metadata['file_path']).group(0)}\nPage: {result.metadata['page']}\nContent:\n{result.page_content}\n"
            for result in reranked_results
        ])
    return results


interface = gr.Interface(
    fn=handle_files_and_query,
    inputs=[
        gr.Textbox(lines=1, label="Enter your search query here..."),
        gr.File(file_count="multiple", type="file", file_types=[".pdf"], label="Upload a file here."),
        gr.Slider(minimum=1, maximum=100, value=50, label="Chunk Overlap"),
        gr.Slider(minimum=64, maximum=512, value=256, label="Tokens Per Chunk (чем больше - тем бОльшие куски книги "
                                                            "сможем находить)"),
        gr.Slider(minimum=1, maximum=1000, value=200, label="BM25 Answers (чем больше - тем больше будем учитывать неявные смысловые сравнения слов)")
    ],
    outputs="text",
    title="Similarity Search for eksmo books"
)

interface.launch()
