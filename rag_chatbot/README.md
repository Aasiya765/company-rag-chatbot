# Company Knowledge Assistant — RAG Chatbot (Streamlit)

A Retrieval-Augmented Generation (RAG) chatbot that answers employee
questions using **only** a company's own local data files. It loads
`.txt` and `.xlsx` files from the `data/` folder at startup, chunks and
indexes them into a FAISS vector store, and answers questions using
**Google's free Gemini API** constrained to the retrieved content.

## Features
- Loads all `.txt` and `.xlsx` files from `data/` automatically at startup
- Chunks content with `RecursiveCharacterTextSplitter`
- Embeds chunks with Gemini's free `embedding-001` model and indexes them in FAISS
- Retrieves the most relevant chunks per question (top-k similarity search)
- Answers with Gemini's free `gemini-1.5-flash` model, strictly from retrieved
  context — refuses to answer if the information isn't in the company data
- Shows source file(s) used for every answer
- Simple Streamlit chat UI with history, index rebuild, and clear-chat controls
- **No paid API required** — works entirely on Google AI Studio's free tier

## Project Structure
```
rag_chatbot/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── .env.example             # Template for your Gemini API key
├── .gitignore
├── README.md
└── data/                    # Company data files (replace with your own)
    ├── company_policies.txt
    ├── faq.txt
    └── employee_data.xlsx   # Multi-sheet sample (Employees, Leave, Benefits)
```

## Setup

1. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a free Gemini API key and add it**
   - Go to https://aistudio.google.com/app/apikey and click "Create API key"
     (no billing required for the free tier).
   ```bash
   cp .env.example .env
   # then edit .env and set GOOGLE_API_KEY=your-key-here
   ```
   Alternatively, you can paste the key directly into the sidebar when the
   app is running (it won't be saved anywhere).

4. **Add your company data**
   Replace the sample files in `data/` with your own `.txt` and `.xlsx`
   files. The app picks up every `.txt` and `.xlsx` file in that folder
   (and subfolders) automatically — no code changes needed.

5. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Open the URL Streamlit prints (usually `http://localhost:8501`).

## How It Works

1. **Load** — `TextLoader` reads `.txt` files; `UnstructuredExcelLoader`
   (mode="elements") reads `.xlsx` files sheet by sheet, keeping tabular
   structure intact.
2. **Chunk** — Documents are split into ~1000-character chunks with 150
   characters of overlap so context isn't lost at chunk boundaries.
3. **Embed & Index** — Chunks are embedded with Gemini's free
   `models/embedding-001` and stored in an in-memory FAISS index (rebuilt
   each time the app restarts or you click "Rebuild index").
4. **Retrieve** — On each question, the top 4 most similar chunks are
   retrieved from FAISS.
5. **Answer** — Gemini's free `gemini-1.5-flash` is given a strict prompt
   instructing it to answer only from the retrieved chunks, and to say so
   explicitly when the answer isn't in the data.

## Customization

| What | Where |
|---|---|
| Chunk size / overlap | `CHUNK_SIZE`, `CHUNK_OVERLAP` in `app.py` |
| Number of retrieved chunks | `TOP_K` in `app.py` |
| Embedding model | `EMBEDDING_MODEL` in `app.py` |
| Chat model | `CHAT_MODEL` in `app.py` |
| Answer refusal behavior / tone | `QA_PROMPT` in `app.py` |
| Persisting the FAISS index to disk | Add `vectorstore.save_local(FAISS_INDEX_DIR)` / `FAISS.load_local(...)` in `build_vectorstore()` for large datasets, so you don't re-embed on every restart |

## Notes on Scaling & Free-Tier Limits
- For large data sets, persist the FAISS index to disk (`save_local` /
  `load_local`) instead of rebuilding it on every app start.
- `unstructured` may require system dependencies (e.g. `libmagic`) on some
  platforms for full Excel/PDF support — see the
  [unstructured docs](https://docs.unstructured.io) if you hit loader
  errors.
- **Gemini free tier has rate limits** (requests per minute/day vary by
  model — check https://ai.google.dev/gemini-api/docs/rate-limits). If you
  have a very large `data/` folder, the initial embedding step may hit the
  rate limit; if that happens, wait a minute and click "Rebuild index"
  again, or split large files into smaller ones.
- `gemini-1.5-flash` and `embedding-001` are used as free, fast defaults;
  swap in `gemini-1.5-pro` or other Gemini models in `app.py` (`CHAT_MODEL`)
  if you need higher quality and don't mind lower free-tier limits.

## Troubleshooting

| Issue | Fix |
|---|---|
| "No documents were found" | Confirm files are directly inside `data/` and have `.txt` or `.xlsx` extensions |
| Excel loader errors | Ensure `unstructured` and `openpyxl` are installed (`pip install -r requirements.txt`); some environments also need `pip install "unstructured[xlsx]"` |
| Answers seem outdated after editing files | Click **Rebuild index** in the sidebar — the index is cached until you rebuild or restart the app |
| `429 Resource has been exhausted` / rate limit errors | You've hit the Gemini free-tier rate limit — wait ~1 minute and retry, or reduce how much data is indexed at once |
| API key errors | Double check `.env` or the sidebar input; confirm the key was created at https://aistudio.google.com/app/apikey |
