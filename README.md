# Insurance Mitra

Insurance Mitra includes a lightweight FastAPI and browser interface for Vercel, plus the original
Streamlit application for local use. Both answer from an educational insurance knowledge base before
asking Google Gemini to respond. It is an educational aid, not a coverage decision or a substitute
for policy wording or qualified advice.

## Requirements

- Python 3.12 or 3.13 for the Vercel app; Python 3.11 or newer for the local Streamlit app.
- A Google AI Studio API key for answer generation. Document loading, indexing, and tests do not
  need a Gemini key.
- Internet access to install dependencies and call Gemini. The local Streamlit app also downloads
   the `all-MiniLM-L6-v2` embedding model on first run.

## Windows PowerShell setup

From the project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-streamlit.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GOOGLE_API_KEY` to your own key. Do not commit `.env`. Alternatively configure
`GOOGLE_API_KEY` as an environment variable or Streamlit secret. For Streamlit secrets, create
`.streamlit\secrets.toml` containing:

```toml
GOOGLE_API_KEY = "your-key"
```

The secrets file is ignored by Git. Never print or share the key. The default `GEMINI_MODEL` is
`gemini-3.8-flash`; change it in `.env` to another model identifier available to your Google AI
project if availability changes. Consult Google's current model catalog before switching models.

## Build and run

Create the PDF source and semantic index (the index is reproducible and is not committed):

```powershell
python build_knowledge_base.py
```

The build command validates that extracted PDF text preserves the Markdown source and fits
inside page bounds before it loads embeddings or rebuilds the FAISS index. It then checks that
the persisted passage count matches the chunks from the extracted source documents.

Force a rebuild:

```powershell
python build_knowledge_base.py --force
```

Start the local Streamlit app:

```powershell
streamlit run app.py
```

Run the Vercel version locally with the production dependencies:

```powershell
pip install -r requirements.txt
python -m uvicorn api.app:app --reload
```

The app also detects changed source files and rebuilds the index as required. The sidebar has a
manual shared-index rebuild action.

## Tests

Install the test and legacy-app dependencies, then run the test suite:

```powershell
pip install -r requirements-dev.txt
pytest -q
```

The tests use deterministic local fake embeddings and mock Gemini generation. They do not call
Gemini or require network access.

## Add and remove private documents

In the sidebar, select PDF files and explicitly choose **Add to this session**. PDF uploads are
limited to 20 MB each, validated and extracted as selectable text, then indexed separately for
the current Streamlit session. They are not written to the shared `vector_store/` or intentionally
saved to disk. When a question uses a private PDF, the retrieved excerpts are sent to the
configured Google Gemini API for answer generation. Use **Remove all session PDFs** to discard
them from the current session, or end the session. Scanned PDFs may have no selectable text and
may require OCR before use. PDF type checking and parsing are not malware scanning; a PDF is not
assumed safe merely because of its extension.

The session is not an isolation boundary against the server operator, hosting provider, process
crashes, backups, or infrastructure-level inspection. Do not upload documents unless you trust
the deployment and are authorized to share their contents. Remove unnecessary identifiers before
uploading. Never upload passwords or one-time codes.

The default educational sources are `data/insurance_knowledge.md` and the generated
`data/insurance_knowledge.pdf`. Replace or extend these only with material you are entitled to use.
Do not put private customer policies in `data/`.

## Streamlit retrieval architecture

1. `document_loader.py` extracts Markdown and PDF text and keeps PDF page numbers.
2. The RAG pipeline chunks text with overlap and retains the source filename and page.
3. Sentence Transformers creates normalized `all-MiniLM-L6-v2` embeddings.
4. FAISS CPU stores vectors in a cosine-similarity index (`IndexFlatIP` over normalized vectors).
5. A SHA-256 fingerprint of source filenames, contents, and embedding/chunk settings lets the app
   reuse an unchanged index or rebuild it after edits or indexing changes. The FAISS index and
   JSON metadata live in `vector_store/`.
6. Each question is embedded and searched before generation. Low-relevance or missing context
   produces an explicit no-evidence response and does not call Gemini.
7. Gemini receives only the question and retrieved passages. The prompt treats passages as
   untrusted data and asks for source citations and uncertainty.

The interface exposes retrieved passages under **RAG Sources**. Relevance threshold and number of
passages can be adjusted with `MIN_RELEVANCE_SCORE` and `RETRIEVAL_TOP_K`. These are retrieval
controls, not guarantees that evidence is complete.

## Vercel deployment

The Vercel entrypoint is configured as `api.app:app` in `pyproject.toml`; Vercel serves the browser
interface from `index.html` and runs the API as a Python function. Import this repository into Vercel
and use the repository root as the project root. Set `GOOGLE_API_KEY` in the Vercel project's
Environment Variables before deploying. `GEMINI_MODEL` is optional and defaults to the value in
`api/app.py`. Never put API keys in frontend code or commit `.env`.

The Vercel version uses lightweight lexical retrieval over `data/insurance_knowledge.md`; it does
not load the local FAISS index or download an embedding model. It is stateless and does not accept
private PDF uploads. Use the local Streamlit version for session-only PDF uploads and semantic FAISS
retrieval. Since the Vercel API can incur Gemini usage costs, configure provider quotas and add
appropriate access controls or rate limiting before sharing a public deployment.

## Important disclaimer

Answers are educational and may omit policy-specific terms, tables, endorsements, or current rules.
They do not guarantee coverage, claim approval, or legal or financial outcomes. Actual coverage
depends on the complete policy wording and applicable regulations. Confirm decisions with your
insurer or a qualified professional.
