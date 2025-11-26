# back_end/workers/celery_app.py
import os, uuid, logging
from celery import Celery
from datetime import datetime
from services.file_processing import extract_text_from_file_bytes
from services.chunker import chunk_text_by_tokens
from services.embeddings import embed_texts
from services.pinecone_client import index
from services.supabase_client import supabase

CELERY_BROKER = os.getenv("CELERY_BROKER_URL")
CELERY_BACKEND = os.getenv("CELERY_RESULT_BACKEND")
celery = Celery("ingest", broker=CELERY_BROKER, backend=CELERY_BACKEND)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@celery.task(bind=True, max_retries=3, acks_late=True)
def ingest_file_task(self, file_path: str, filename: str, file_bytes: bytes = None, user_id: str = None, file_id: str = None):
    """
    file_path: path in Supabase bucket (if file_bytes is None)
    """
    try:
        file_id = file_id or str(uuid.uuid4())

        # Fetch bytes if not provided
        content_bytes = file_bytes
        if content_bytes is None:
            # supabase returns http response object; use storage.download
            bucket = supabase.storage.from_(os.getenv("SUPABASE_BUCKET"))
            dl = bucket.download(file_path)
            if hasattr(dl, "read"):
                content_bytes = dl.read()
            else:
                content_bytes = dl

        if not content_bytes:
            raise ValueError("No content to ingest")

        # Extract text
        text = extract_text_from_file_bytes(filename, content_bytes)
        if not text or not text.strip():
            logger.warning("No text extracted for %s", filename)
            return {"status": "no_text"}

        # Chunk
        chunks_meta = chunk_text_by_tokens(text)
        texts = [c[0] for c in chunks_meta]
        if not texts:
            return {"status": "no_chunks"}

        # Batch embed
        vectors = embed_texts(texts)

        # Build upserts in batches for Pinecone
        upserts = []
        for (chunk_text, start, end), vec in zip(chunks_meta, vectors):
            chunk_id = f"{file_id}:{start}-{end}"
            meta = {
                "filename": filename,
                "file_id": file_id,
                "text": chunk_text[:2000],  # store snippet (limit metadata size)
                "created_at": datetime.utcnow().isoformat(),
                "user_id": user_id
            }
            upserts.append((chunk_id, vec, meta))

        # Upsert into pinecone in batches
        batch_size = 100
        for i in range(0, len(upserts), batch_size):
            batch = upserts[i:i+batch_size]
            ids = [u[0] for u in batch]
            vecs = [u[1] for u in batch]
            metas = [u[2] for u in batch]
            index.upsert(vectors=list(zip(ids, vecs, metas)))

        logger.info("Ingested %d chunks for %s", len(upserts), filename)
        return {"status": "ok", "inserted": len(upserts)}

    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise self.retry(exc=exc, countdown=min(60 * (2 ** self.request.retries), 300))
