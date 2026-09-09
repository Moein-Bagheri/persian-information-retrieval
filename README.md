# Persian Information Retrieval

A two-phase text-search project for a Persian document collection. It builds an **inverted index**, supports Boolean retrieval, and adds **TF-IDF and BM25 ranking** in the second phase.

**Python · Hazm · inverted indexing · Boolean search · TF-IDF · BM25**

## Project phases

| Folder | Contents |
| --- | --- |
| [`01-boolean-search/`](01-boolean-search/) | Persian preprocessing and Boolean retrieval |
| [`02-ranked-search/`](02-ranked-search/) | The extended engine with TF-IDF and BM25 scores |
| [`docs/`](docs/) | Original Persian implementation reports |

Each phase retains its original `Main.py` and local JSON corpus. The small corpus is present in both folders because the independent entry points expect it beside the script.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Push-Location 01-boolean-search
python Main.py
Pop-Location

Push-Location 02-ranked-search
python Main.py
Pop-Location
```

Run one phase at a time and follow its console interaction. Preprocessing uses Hazm normalization, tokenization, stopword removal and lemmatization. The ranking calculations are implemented in the project source.

## Documentation

- [Boolean-search report in Persian](docs/boolean-search-report-fa.pdf)
- [Ranked-search report in Persian](docs/ranked-search-report-fa.pdf)

The reports credit **Moeinreza Bagheri and Somayeh Heydari**. This is a university information-retrieval project; course data and implementation reports are retained as project material.
