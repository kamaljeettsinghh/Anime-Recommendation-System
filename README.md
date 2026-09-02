# Similar — Anime Recommender (Streamlit, Html)

A content-based anime recommender: pick a title, get the 10 closest matches by
genre and format (TV/Movie/OVA/…), using TF-IDF + cosine similarity. Includes
a second tab with basic EDA on the dataset.

## Files

- `app.py` — the Streamlit app
- `anime.csv` — the dataset the app reads (must stay next to `app.py`)
- `requirements.txt` — Python dependencies
- `anime_recommender.html` - HTML Deployed

## Processed Dataset Description:

- Unique ID of each anime
- Anime title
- Anime broadcast type, such as TV, OVA, etc
- anime genre
- The number of episodes of each anime
- The average rating for each anime compared to the number of users who gave ratings


## Tasks:
-Data Preprocessing:
-Feature Extraction:
-Decide on the features that will be used for computing similarity (e.g., genres, user ratings).
-Convert categorical features into numerical representations if necessary.




