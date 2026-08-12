# YouTube RAG Chat — Chrome Extension

Turns your notebook's RAG pipeline into a daily-use Chrome extension:
popup UI → FastAPI backend → transcript fetch + FAISS retrieval + LLM.

## Structure
```
youtube-rag-extension/
  backend/
    main.py              # FastAPI wrapper around your notebook's pipeline
    requirements.txt
    .env.example          # rename to .env, add your HF token
  extension/
    manifest.json
    popup.html
    popup.js
    icons/                # placeholder icons, swap with your own
```

## 1. Run the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env with your real HUGGINGFACEHUB_API_TOKEN
uvicorn main:app --reload --port 8000
```

Test it works before touching the extension:
```bash
curl -X POST http://localhost:8000/index -H "Content-Type: application/json" \
  -d '{"video_id": "Gfr50f6ZBvo"}'

curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"video_id": "Gfr50f6ZBvo", "question": "What is this video about?"}'
```

## 2. Load the extension

1. Go to `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**, select the `extension/` folder
4. Pin the extension, open any YouTube video, click the icon, ask a question

## 3. Daily-use notes

- **First question on a video is slow** — it fetches the transcript, chunks, and embeds. Subsequent questions on the same video reuse the cached retriever (as long as the backend process hasn't restarted).
- **Backend must be running** for the extension to work locally (`uvicorn main:app --reload --port 8000` needs to stay up). For real daily use without keeping your laptop's terminal open, deploy the backend (see below) and point `API_BASE` in `popup.js` at that URL.
- **Videos with no captions** will return a clear error — your notebook's `TranscriptsDisabled` handling is preserved in `main.py`.

## 4. Deploying the backend (optional, for "always on" use)

Any small always-on host works: Render, Railway, Fly.io, a cheap VM.
Steps are the same regardless of host:
1. Push `backend/` to a git repo
2. Set `HUGGINGFACEHUB_API_TOKEN` as an environment variable on the host (not in code)
3. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Once deployed, update `API_BASE` in `extension/popup.js` to the HTTPS URL
5. Update `host_permissions` in `manifest.json` to that same domain instead of `localhost:8000`
6. Reload the extension in `chrome://extensions`

## 5. Known gaps to harden later

- In-memory cache means indexed videos are forgotten on backend restart — persist with `FAISS.save_local()` per video_id if that matters to you.
- CORS is wide open (`*`) for local dev — restrict to your extension's ID once deployed.
- No auth on the API — fine for personal use, but add a shared-secret header if you deploy it publicly.
