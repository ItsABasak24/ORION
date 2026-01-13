import logging
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai
import google.generativeai as genai
import dotenv
import os

dotenv.load_dotenv()

try:
    genai.configure(api_key=os.getenv("API_KEY"))
    print("API key configuration successful.")
except Exception as e:
    print(f"API key configuration failed: {e}")
    
model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={
        "temperature":0.2,
        "top_p":0.8,
        "response_mime_type":"text/plain"
    }
)

print("Model intuitions Done!")


def geminiAi(task: str | None) -> str:
    """
    Generate content using Gemini AI model based on the provided task.
    Args:
        task (str): The task or prompt for content generation.
    Returns:
        str: Generated content or an error message.
    """
    if not task:
        return "No task provided."
    try:
        response = model.generate_content(task)
        return f"{response.text}"
    except Exception as e:
        return f"Exception: {str(e)}"


# Configure basic logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



def ragEngine(pdf_path: str, api_key: str):
    # Embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Load PDF & build FAISS index
    logger.debug("Initializing RAGEngine with pdf_path=%s, model=%s", pdf_path)
    chunks, pages = load_pdf(pdf_path)
    logger.debug("Loaded %d chunks from %d pages", len(chunks), len(set(pages)))
    index = build_index(chunks, embedder)
    logger.debug("FAISS index built. Index nlist/count: %s", getattr(index, 'ntotal', 'unknown'))

    # Chat memory
    return{
        "embedder": embedder,
        "index": index,
        "chunks": chunks,
        "pages": pages,
        "chat_history": []
    }

    # ------------------ PDF LOADING ------------------
def load_pdf(path):
    logger.debug("Loading PDF from path=%s", path)
    reader = PdfReader(path)
    chunks, pages = [], []

    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
        except Exception:
            logger.exception("Failed to extract text from page %s", page_no)
            continue

        if not text:
            logger.debug("No text extracted from page %s", page_no)
            continue

        for para in text.split("\n\n"):
            para = para.strip()
            if len(para) > 20:
                chunks.append(para)
                pages.append(page_no)

    logger.debug("PDF load complete: %d chunks found", len(chunks))
    return chunks, pages

    # ------------------ FAISS INDEX ------------------
def build_index(chunks, embedder):
    logger.debug("Building FAISS index for %d chunks", len(chunks))
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    logger.debug("Embeddings shape: %s", getattr(embeddings, 'shape', None))
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    try:
        n = index.ntotal
    except Exception:
        n = 'unknown'
    logger.debug("Index built and populated. ntotal=%s", n)
    return index

    # ------------------ SEMANTIC SEARCH ------------------
def search(query, embedder, index, chunks, pages, k=3):
    logger.debug("Search called with query=%s, k=%s", query, k)
    q_embed = embedder.encode([query], convert_to_numpy=True)
    logger.debug("Query embedding shape: %s", getattr(q_embed, 'shape', None))
    _, idxs = index.search(q_embed, k)
    logger.debug("Search returned indices: %s", idxs)
    results = [(chunks[i],pages[i]) for i in idxs[0]]
    logger.debug("Returning %d search results", len(results))
    logger.info("Search results: %s", results)
    return results

    # ------------------ CHAT WITH MEMORY + CITATIONS ------------------
    
def chat(query, ragState):
    embedder = ragState['embedder']
    index = ragState['index']
    chunks = ragState['chunks']
    pages = ragState['pages']
    chat_history = ragState['chat_history']
    # Normalize query
    query_lower = query.strip().lower()

    greetings = [
        "hi", "hello", "hey", "good morning",
        "good evening", "good afternoon", "how are you", "good night"
        ]

    # ------------------ GREETING HANDLING ------------------
    if any(greet in query_lower for greet in greetings):
        answer = (
                "Hello! 👋 I’m ORION, your Enterprise Intelligence Assistant.\n"
                "How can I assist you with your company's knowledge today?"
                
            )
        logger.debug("Greeting detected in query: %s", query)
        # Save conversation
        chat_history.append(f"User: {query}")
        chat_history.append(f"ORION: {answer}")
        logger.debug("Responded to greeting.")
        # # ❌ No citations
        return answer, []

    # ------------------ DOCUMENT-BASED QUERY ------------------
    results = search(query, embedder, index, chunks, pages)
    logger.debug("Document-based query; %d contextual chunks retrieved", len(results))

    context = "\n\n".join(
        [f"(Page {page}) {chunk}" for chunk, page in results]
        )

    chat_history.append(f"User: {query}")

    prompt = f"""
    You are ORION, an Enterprise Intelligence Assistant.

    Conversation History:
    {chr(10).join(chat_history[-5:])}

    Context from Company Knowledge Base:
    {context}

    Instructions:
    - Answer professionally and concisely
    - Use only the provided context
- Cite page numbers where relevant
"""
    try:
        logger.debug("Sending prompt to model")
        logger.debug(f"{prompt}")
        response = geminiAi(task=prompt)
        answer = response
        logger.debug("Model response received (length=%d)", len(answer) if answer else 0)

        # Normal case → citations allowed
        chat_history.append(f"ORION: {answer}")
        return answer, results

    except Exception:
        # Gemini failed → NO citations
        answer = (
            "⚠️ The AI service is temporarily unavailable.\n\n"
            "Please use the search feature or try again later."
        )

        chat_history.append(f"ORION: {answer}")
        logger.exception("Model call failed for query: %s", query)

        # ❌ RETURN EMPTY SOURCES
        return answer, []
