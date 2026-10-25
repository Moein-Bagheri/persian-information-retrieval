# Persian search engine

**A search engine for Persian text, built from the ground up.** Hazm handles the language; the inverted index, Boolean search, TF-IDF and BM25 are written by hand, with no ranking libraries.

*Python · Hazm* · Built with Somayeh Heydari · Reports in Persian: [Boolean search](docs/boolean-search-report-fa.pdf) · [Ranked search](docs/ranked-search-report-fa.pdf)

## What we did

- Indexed 100 Persian thesis records (title, abstract and keywords) from IranDoc's Ganj, retrieved for the query «هوش مصنوعی» (artificial intelligence).
- Preprocessed with Hazm: normalization, tokenization, stopword removal and lemmatization.
- **Phase 1:** an inverted index with Boolean AND and OR search.
- **Phase 2:** TF, IDF, TF-IDF and BM25 (k1 = 1.2) implemented by hand, plus an interactive menu to search either way.

## What we found

Two queries, two outcomes. On «مسئولیت مدنی» (civil liability), TF-IDF and BM25 agree on the top four documents. On «پزشکی درمان» (medicine, treatment), they don't:

| Rank | TF-IDF | BM25 |
| ---: | --- | --- |
| 1 | doc 37 | doc 6 |
| 2 | doc 3 | doc 33 |
| 3 | doc 6 | doc 15 |
| 4 | doc 15 | doc 37 |

- **Length changes the answer.** BM25 discounts long documents, so TF-IDF's first result drops to fourth.
- **Edge cases behave.** Unknown words and stopword-only queries return nothing, rather than noise.

## Try it

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Push-Location 02-ranked-search
python Main.py
Pop-Location
```

The menu offers Boolean AND, Boolean OR, TF-IDF and BM25. Phase 1 runs the same way from `01-boolean-search/`.

## Layout

| Path | What's there |
| --- | --- |
| [`01-boolean-search/`](01-boolean-search/) | Preprocessing, inverted index and Boolean search |
| [`02-ranked-search/`](02-ranked-search/) | The extended engine with TF-IDF and BM25 |
| [`docs/`](docs/) | Both phase reports, in Persian |

Each phase keeps its own copy of the corpus beside its script.

Information retrieval course project with Somayeh Heydari, University of Isfahan, 2026.
