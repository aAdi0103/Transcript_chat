import os
import time
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import (
    HuggingFaceEndpointEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="YouTube RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_video_cache: Dict[str, dict] = {}

PROMPT = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      Use the conversation history only to resolve references in the current
      question (for example, "it", "that", or "the second point"). Do not
      treat chat history as factual evidence; all factual claims must be
      supported by the transcript context.

      Keep your answer under 450 words. Be complete but concise - fully
      finish your last sentence and do not leave the answer trailing off
      or unfinished. If the topic is broad, pick the 3-4 most important
      points rather than trying to enumerate everything covered.

      {context}
      Conversation history:
      {history}
      Question: {question}
    """,
    input_variables=["context", "history", "question"],
)

# Used automatically as a fallback if the first answer comes back truncated.
# Much tighter budget so it has no realistic way of running out of room again.
PROMPT_STRICT = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      Use the conversation history only to resolve references in the current
      question. Do not treat chat history as factual evidence.

      Give a SHORT answer: under 200 words, at most 3 key points.
      You MUST end with a complete sentence - never leave a thought unfinished.
      Prioritize finishing over covering more ground.

      {context}
      Conversation history:
      {history}
      Question: {question}
    """,
    input_variables=["context", "history", "question"],
)


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def format_history(history: List["ConversationTurn"]) -> str:
    if not history:
        return "No previous conversation."
    return "\n".join(
        f"User: {turn.question}\nAssistant: {turn.answer}" for turn in history
    )


def fetch_transcript_text(video_id: str) -> str:

    api = YouTubeTranscriptApi()

    try:
        transcript_list = api.list(video_id)
    except TranscriptsDisabled:
        raise HTTPException(status_code=422, detail="Captions are disabled for this video.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to list transcripts: {e}")

    transcript_obj = None

    # 1. Try to find an English transcript directly.
    try:
        transcript_obj = transcript_list.find_transcript(["en"])
    except NoTranscriptFound:
        pass

    # 2. No English track — take the first available transcript in any language.
    if transcript_obj is None:
        available = list(transcript_list)
        if not available:
            raise HTTPException(status_code=422, detail="No transcript found for this video.")
        transcript_obj = available[0]

        # Try translating it to English.
        if transcript_obj.is_translatable:
            try:
                transcript_obj = transcript_obj.translate("en")
            except Exception:
                # Translation failed - proceed with the original language.
                # The RAG pipeline will still work, just answers may reflect
                # the original language's wording in retrieved chunks.
                pass

    try:
        fetched = transcript_obj.fetch()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch transcript: {e}")

    transcript = " ".join(chunk.text for chunk in fetched)
    if not transcript.strip():
        raise HTTPException(status_code=422, detail="Transcript is empty.")

    return transcript


def build_retriever_and_llm(video_id: str):
    """Fetch transcript, chunk, embed, and return a (retriever, llm) pair for a video."""
    transcript = fetch_transcript_text(video_id)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    
    embeddings = HuggingFaceEndpointEmbeddings(model="ibm-granite/granite-embedding-97m-multilingual-r2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = HuggingFaceEndpoint(
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
        task="text-generation",
        max_new_tokens=1536,
        temperature=0.3,
    )
    llm = ChatHuggingFace(llm=llm)

    return retriever, llm, len(transcript)


def run_query(retriever, llm, question: str, history: List["ConversationTurn"], strict: bool = False) -> str:
    """Build the retrieval+prompt+LLM chain on demand and invoke it.

    strict=True swaps in a much tighter prompt (shorter target length),
    used as an automatic fallback when the normal answer comes back
    truncated.
    """
    prompt = PROMPT_STRICT if strict else PROMPT
    formatted_history = format_history(history)
    # Including recent turns in the retrieval query lets follow-ups such as
    # "What does it mean?" retrieve chunks about the subject mentioned earlier.
    retrieval_query = f"Conversation history:\n{formatted_history}\n\nCurrent question: {question}"
    parallel_chain = RunnableParallel(
        {
            "context": RunnableLambda(lambda _: retrieval_query) | retriever | RunnableLambda(format_docs),
            "history": RunnableLambda(lambda _: formatted_history),
            "question": RunnableLambda(lambda _: question),
        }
    )
    chain = parallel_chain | prompt | llm | StrOutputParser()
    return chain.invoke({})


def run_query_with_retry(retriever, llm, question: str, history: List["ConversationTurn"], strict: bool = False, attempts: int = 3) -> str:
    """Wraps run_query with retry-with-backoff for transient HuggingFace
    errors (rate limits, cold-start timeouts, momentary 5xx from the
    Inference API). These show up as generic exceptions from run_query,
    not as empty/truncated answers, so they need their own handling.
    """
    last_error = None
    for attempt in range(attempts):
        try:
            return run_query(retriever, llm, question, history, strict=strict)
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            is_transient = any(
                token in msg
                for token in ["429", "rate limit", "timeout", "timed out", "503", "overloaded", "temporarily"]
            )
            if not is_transient or attempt == attempts - 1:
                raise
            # Exponential backoff: 2s, 4s, ...
            time.sleep(2 * (attempt + 1))
    raise last_error


def looks_truncated(text: str) -> bool:
    """Heuristic: does this answer look like it was cut off mid-thought?

    A properly finished answer ends with sentence-ending punctuation
    (optionally followed by a closing quote/bracket). Anything else -
    trailing off after a comma, an open quote, mid-word, etc. - almost
    always means the token budget ran out before the model finished.
    """
    t = text.strip()
    if not t:
        return True
    if len(t) < 20:
        # Very short answers ("I don't know.") are rarely truncations.
        return False

    tail = t.rstrip("\"')]")
    return not tail.endswith((".", "!", "?"))



class IndexRequest(BaseModel):
    video_id: str


class ConversationTurn(BaseModel):
    question: str
    answer: str


class QueryRequest(BaseModel):
    video_id: str
    question: str
    history: List[ConversationTurn] = []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index")
def index_video(req: IndexRequest):
    
    video_id = req.video_id.strip()
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required.")

    if video_id in _video_cache:
        return {"video_id": video_id, "cached": True}

    retriever, llm, transcript_len = build_retriever_and_llm(video_id)
    _video_cache[video_id] = {"retriever": retriever, "llm": llm}
    return {"video_id": video_id, "cached": False, "transcript_chars": transcript_len}


@app.post("/query")
def query_video(req: QueryRequest):
    """Ask a question about an already-indexed video."""
    video_id = req.video_id.strip()
    question = req.question.strip()

    if not video_id or not question:
        raise HTTPException(status_code=400, detail="video_id and question are required.")

    if video_id not in _video_cache:
        # Auto-index on first query so the extension doesn't need two calls.
        retriever, llm, _ = build_retriever_and_llm(video_id)
        _video_cache[video_id] = {"retriever": retriever, "llm": llm}

    retriever = _video_cache[video_id]["retriever"]
    llm = _video_cache[video_id]["llm"]

    try:
        # Keep the request bounded even if a client sends an oversized history.
        history = req.history[-6:]
        answer = run_query_with_retry(retriever, llm, question, history)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"The model backend didn't respond successfully after retries: {e}",
        )

    # Occasionally the model returns an empty string (e.g. it spent its whole
    # token budget on internal reasoning before writing anything visible).
    # Retry once before giving up, since this is usually transient.
    if not answer or not answer.strip():
        try:
            answer = run_query_with_retry(retriever, llm, question, history)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM error on retry: {e}")

    if not answer or not answer.strip():
        raise HTTPException(
            status_code=502,
            detail="The model returned an empty response. Please try asking again.",
        )

    # If the answer looks cut off mid-thought (ran out of token budget),
    # automatically retry once with a much tighter prompt that has no
    # realistic way of running out of room again.
    if looks_truncated(answer):
        try:
            stricter_answer = run_query_with_retry(retriever, llm, question, history, strict=True)
            if stricter_answer and stricter_answer.strip() and not looks_truncated(stricter_answer):
                answer = stricter_answer
        except Exception:
            # If the retry itself fails, just fall back to the original
            # (possibly truncated) answer rather than erroring the request.
            pass

    return {"video_id": video_id, "question": question, "answer": answer}
