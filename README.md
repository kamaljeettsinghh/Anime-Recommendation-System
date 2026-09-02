# Similar — Anime Recommender (Streamlit)

A content-based anime recommender: pick a title, get the 10 closest matches by
genre and format (TV/Movie/OVA/…), using TF-IDF + cosine similarity. Includes
a second tab with basic EDA on the dataset.

## Files

- `app.py` — the Streamlit app
- `anime.csv` — the dataset the app reads (must stay next to `app.py`)
- `requirements.txt` — Python dependencies

## Run it locally

Requires Python 3.9+.

```bash
cd anime_streamlit_app
pip install -r requirements.txt
streamlit run app.py
```


## Deploy 

This environment has no internet access, so it can't push code to GitHub or
stand up a live URL for you — but deploying yourself takes about five
minutes:
