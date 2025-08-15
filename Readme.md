# PDF RAG – Chat with Your PDF

A lightweight app that lets you upload a PDF, index its content with embeddings, and ask natural language questions.
The AI retrieves the most relevant chunks and answers with a reference to the original page.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/kiwan8/rag_pdf
cd rag_pdf

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_second_api_here
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## Workflow

1. **Upload** a PDF.
2. Click **Process** to index its content.
3. **Ask** a question in plain English.
4. Get an **answer** with a snippet from the PDF.

---

## Key Settings

| Setting         | Description                                  | Default                                   |
| --------------- | -------------------------------------------- | ----------------------------------------- |
| `k`             | Number of chunks retrieved per query         | 6                                         |
| `model`         | AI model used for responses                  | `gpt-4o-mini`                             |
| `temperature`   | Creativity level (0 = factual, 1 = creative) | 0.5                                       |
| `chunk_size`    | Max characters per chunk                     | 500                                       |
| `chunk_overlap` | Overlap between chunks to preserve context   | 50                                        |
| `embedding`     | Embedding model for vectorization            | `sentence-transformers/all-mpnet-base-v2` |

---

## Notes

* Switch `device` to `"cuda"` for GPU acceleration.
* Large PDFs may take longer to process.
* Built with **Streamlit**, **LangChain**, and **Chroma**.

