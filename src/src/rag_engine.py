from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_FILE = BASE_DIR / "knowledge" / "ecosystem_knowledge.txt"

VECTOR_DB_DIR = BASE_DIR / "knowledge" / "chroma_db"


# ============================================================
# LOAD KNOWLEDGE
# ============================================================

def load_knowledge():

    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge file not found:\n{KNOWLEDGE_FILE}"
        )

    loader = TextLoader(
        str(KNOWLEDGE_FILE),
        encoding="utf-8"
    )

    documents = loader.load()

    return documents


# ============================================================
# SPLIT KNOWLEDGE INTO CHUNKS
# ============================================================

def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# ============================================================
# CREATE / LOAD VECTOR DATABASE
# ============================================================

def create_vector_database():

    embeddings = create_embeddings()

    # If vector database already exists, load it.
    if VECTOR_DB_DIR.exists() and any(VECTOR_DB_DIR.iterdir()):

        vector_db = Chroma(
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=embeddings
        )

        return vector_db

    # Otherwise create a new database.
    documents = load_knowledge()

    chunks = create_chunks(documents)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_DIR)
    )

    return vector_db


# ============================================================
# RETRIEVE RELEVANT KNOWLEDGE
# ============================================================

def retrieve_context(query, k=3):

    vector_db = create_vector_database()

    results = vector_db.similarity_search(
        query,
        k=k
    )

    context = []

    for document in results:
        context.append(document.page_content)

    return context


# ============================================================
# TEST RAG ENGINE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("🌱 AI ECOSYSTEM RAG ENGINE")
    print("=" * 60)

    query = "How can AI support forest conservation?"

    print("\n🔎 Query:")
    print(query)

    results = retrieve_context(query)

    print("\n📚 Retrieved Knowledge:\n")

    for number, result in enumerate(results, start=1):

        print(f"--- Result {number} ---")
        print(result)
        print()

    print("=" * 60)
    print("✅ RAG retrieval test completed")
    print("=" * 60)