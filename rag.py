import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

# Load API Keys

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found in .env file")


# Extract Video ID

def extract_video_id(url: str):
    """
    Extract YouTube video ID from URL.
    """

    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]

    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    else:
        return url


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# RAG Function

def get_answer(youtube_url, question):

    video_id = extract_video_id(youtube_url)

    # Fetch transcript
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
    except Exception as e:
        return f"Unable to fetch the transcript. Details: {e}"

    transcript_text = " ".join(
        snippet.text for snippet in transcript
    )

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([transcript_text])

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    faiss_retriever = vector_store.as_retriever(
        search_kwargs={"k": 15}
    )

    # BM25
    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        k=15
    )

    # Hybrid Retriever
    retriever = EnsembleRetriever(
        retrievers=[
            faiss_retriever,
            bm25_retriever
        ],
        weights=[0.6, 0.4]
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    def rerank_documents(query, documents, top_k=4):
        
        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        scores = reranker.predict(pairs)

        ranked_docs = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True
        )

        return [doc for _, doc in ranked_docs[:top_k]]

    # Prompt
    prompt = PromptTemplate(
        template="""
You are a helpful AI assistant.

Answer ONLY from the provided transcript context.

If the answer is not available in the transcript,
reply with:

"I couldn't find this information in the video."

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

    parser = StrOutputParser()

    chain1 = (
        RunnableParallel(
            {
                "context": (retriever | 
                            RunnableLambda(lambda docs: rerank_documents(question, docs, top_k=4)) | 
                            RunnableLambda(format_docs)
                            ),
                "question": RunnablePassthrough(),
            }
        )
    )

    chain2 = prompt | llm | parser

    chain = chain1 | chain2

    answer = chain.invoke(question)

    return answer


#  Testing

if __name__ == "__main__":

    youtube_url = "https://www.youtube.com/watch?v=kCc8FmEb1nY"

    question = "What is self attention?"

    answer = get_answer(youtube_url, question)

    print(answer)