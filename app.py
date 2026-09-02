import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Similar — Anime Recommender", page_icon="似", layout="wide")

# ---------------------------------------------------------------------------
# Data loading & model building (cached so this only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading anime data…")
def load_data():
    df = pd.read_csv("anime.csv")
    df = df.dropna(subset=["genre"]).reset_index(drop=True)
    df["type"] = df["type"].fillna("Unknown")
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["tag"] = df["genre"].str.replace(", ", " ", regex=False) + " " + df["type"]
    return df


@st.cache_resource(show_spinner="Building the similarity model…")
def build_model(df: pd.DataFrame):
    tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z\-]+")
    tfidf_matrix = tfidf.fit_transform(df["tag"])
    sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, sim


df = load_data()
tfidf, cosine_sim = build_model(df)
title_to_index = pd.Series(df.index, index=df["name"]).drop_duplicates()


def recommend(title: str, n: int = 10) -> pd.DataFrame:
    idx = title_to_index[title]
    sims = list(enumerate(cosine_sim[idx]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    sims = [s for s in sims if s[0] != idx][:n]
    result_idx = [s[0] for s in sims]
    scores = [s[1] for s in sims]
    out = df.iloc[result_idx][["name", "genre", "type", "rating", "members"]].copy()
    out["match"] = [round(s * 100, 1) for s in scores]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("似 Similar — an anime recommender")
st.caption(
    "Content-based recommendations from genre and format (TV, Movie, OVA…), "
    "using TF-IDF + cosine similarity. No account or watch history needed."
)

tab_recommend, tab_explore = st.tabs(["Get recommendations", "Explore the data"])

# ---------- Tab 1: Recommender ----------
with tab_recommend:
    col_search, col_n = st.columns([3, 1])
    with col_search:
        default_idx = 0
        titles = df["name"].tolist()
        preset = "Naruto" if "Naruto" in titles else titles[0]
        selected = st.selectbox(
            "Pick an anime you've watched",
            options=titles,
            index=titles.index(preset),
        )
    with col_n:
        n = st.slider("How many recommendations?", min_value=5, max_value=20, value=10)

    base = df[df["name"] == selected].iloc[0]
    st.markdown(f"### Because you watched *{base['name']}*")
    st.markdown(
        f"**{base['type']}** &nbsp;·&nbsp; ★ {base['rating']:.2f} &nbsp;·&nbsp; "
        f"{base['members']:,} members  \n"
        f"Genres: {base['genre']}"
    )

    recs = recommend(selected, n=n)

    if recs.empty:
        st.info("No close genre matches found for this title.")
    else:
        st.markdown("#### Top matches")
        for i, row in recs.iterrows():
            c1, c2, c3 = st.columns([0.5, 5, 1.3])
            c1.markdown(f"**{i + 1}**")
            with c2:
                st.markdown(f"**{row['name']}**")
                st.caption(f"{row['type']} · ★ {row['rating']:.2f} · {row['genre']}")
            with c3:
                st.markdown(f"**{row['match']}%** match")
                st.progress(min(1.0, row["match"] / 100))

        st.markdown("#### Match scores")
        chart = (
            alt.Chart(recs)
            .mark_bar(color="#c1442d")
            .encode(
                x=alt.X("match:Q", title="Similarity (%)"),
                y=alt.Y("name:N", sort="-x", title=""),
                tooltip=["name", "genre", "type", "match"],
            )
            .properties(height=32 * len(recs))
        )
        st.altair_chart(chart, use_container_width=True)

# ---------- Tab 2: EDA ----------
with tab_explore:
    st.markdown("### Dataset overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Anime", f"{len(df):,}")
    m2.metric("Avg rating", f"{df['rating'].mean():.2f}")
    m3.metric("Types", df["type"].nunique())
    m4.metric("Distinct genre tokens", len(tfidf.get_feature_names_out()))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Anime count by type**")
        type_counts = df["type"].value_counts().reset_index()
        type_counts.columns = ["type", "count"]
        st.altair_chart(
            alt.Chart(type_counts)
            .mark_bar(color="#c1442d")
            .encode(x="count:Q", y=alt.Y("type:N", sort="-x")),
            use_container_width=True,
        )
    with col2:
        st.markdown("**Rating distribution**")
        st.altair_chart(
            alt.Chart(df)
            .mark_bar(color="#c9a227")
            .encode(x=alt.X("rating:Q", bin=alt.Bin(maxbins=30)), y="count()"),
            use_container_width=True,
        )

    st.markdown("**Top 15 genres**")
    top_genres = (
        df["genre"].str.split(", ").explode().value_counts().head(15).reset_index()
    )
    top_genres.columns = ["genre", "count"]
    st.altair_chart(
        alt.Chart(top_genres)
        .mark_bar(color="#7a2c1e")
        .encode(x="count:Q", y=alt.Y("genre:N", sort="-x")),
        use_container_width=True,
    )

    st.markdown("**Rating vs. popularity**")
    st.altair_chart(
        alt.Chart(df)
        .mark_circle(opacity=0.35, color="#8c8377")
        .encode(
            x=alt.X("members:Q", scale=alt.Scale(type="log"), title="Members (log scale)"),
            y=alt.Y("rating:Q"),
            tooltip=["name", "type", "rating", "members"],
        ),
        use_container_width=True,
    )
