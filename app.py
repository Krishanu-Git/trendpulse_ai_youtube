# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langdetect import detect, LangDetectException
import traceback
import re
import logging

# Project modules
from src.data_collection.collector import YouTubeCollector
from src.data_collection.reddit_collector import RedditCollector
from src.database.db_handler import MongoHandler
from src.processing.preprocessor import clean_text_multilingual
from src.analysis.sentiment import get_sentiment
from src.analysis.trends import get_trends_over_time
import config

# simple logging to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

st.set_page_config(page_title="TrendPulse AI: YouTube + Reddit", layout="wide", page_icon="📺", initial_sidebar_state="expanded")


def detect_language(text):
    try:
        return detect(text)
    except (LangDetectException, TypeError):
        return "unknown"


# ---------------- defensive helpers ----------------
def make_unique_columns(columns):
    seen = {}
    out = []
    for c in columns:
        key = c if pd.notna(c) else ""
        if key in seen:
            seen[key] += 1
            out.append(f"{key}__{seen[key]}")
        else:
            seen[key] = 0
            out.append(key)
    return out


def ensure_unique_columns(df):
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    cols = list(df.columns)
    if len(cols) == len(set(cols)):
        return df
    df2 = df.copy()
    df2.columns = make_unique_columns(cols)
    return df2


def safe_concat_rows(frames):
    frames = [ensure_unique_columns(f) for f in frames if f is not None and isinstance(f, pd.DataFrame)]
    if not frames:
        return pd.DataFrame()
    try:
        return pd.concat(frames, ignore_index=True, sort=False)
    except Exception:
        union = []
        for f in frames:
            for c in f.columns:
                if c not in union:
                    union.append(c)
        reindexed = [f.reindex(columns=union) for f in frames]
        return pd.concat(reindexed, ignore_index=True, sort=False)


def _find_best_time_column(df: pd.DataFrame):
    """
    Return the best-matching column name in df to use as timestamp.
    Preference order: exact known names, then any column containing key words.
    """
    if df is None or df.empty:
        return None

    preferred = ["comment_published_at", "publishedAt", "published_at", "created_utc", "created_at", "created", "comment_published", "comment_publishedAt", "utc_datetime", "comment_date", "timestamp"]
    cols_lower = {c: c.lower() for c in df.columns}

    # exact/preferred matches (case-insensitive)
    for p in preferred:
        for orig, low in cols_lower.items():
            if low == p.lower():
                return orig

    # fallback: any column that contains these tokens
    tokens = ["created", "utc", "published", "time", "date", "timestamp"]
    for token in tokens:
        for orig, low in cols_lower.items():
            if token in low:
                return orig

    return None


def _ensure_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust timestamp detection:
    - look for exact known names, else search by tokens in column names
    - coerce values to datetime (errors -> NaT)
    - returns df with 'comment_published_at' column (datetime64[ns, UTC if tz present])
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    best = _find_best_time_column(df)
    if best:
        try:
            # convert with pandas (handles ISO strings and epoch numbers)
            df["comment_published_at"] = pd.to_datetime(df[best], errors="coerce", utc=True)
            logging.info(f"_ensure_timestamp_col: using column '{best}' for timestamps (converted).")
        except Exception as e:
            logging.warning(f"_ensure_timestamp_col: conversion using '{best}' failed: {e}; trying string coercion.")
            try:
                df["comment_published_at"] = pd.to_datetime(df[best].astype(str), errors="coerce", utc=True)
                logging.info(f"_ensure_timestamp_col: conversion using '{best}' (string) succeeded.")
            except Exception as e2:
                logging.error(f"_ensure_timestamp_col: final conversion failed for '{best}': {e2}")
                df["comment_published_at"] = pd.to_datetime(pd.Series([pd.NaT] * len(df)))
    else:
        logging.info("_ensure_timestamp_col: no timestamp-like column found; creating empty NaT column.")
        df["comment_published_at"] = pd.to_datetime(pd.Series([pd.NaT] * len(df)))
    return df


# ---------------- reddit normalization ----------------
def _normalize_reddit_posts(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns for reddit posts DataFrame coming from the collector.
    Prefer keeping any provided created_utc/created_at/created rather than overwriting.
    """
    if posts_df is None or posts_df.empty:
        return pd.DataFrame()
    df = posts_df.copy()
    df["id"] = df.get("id")
    df["title"] = df.get("title", "")
    df["selftext"] = df.get("selftext", df.get("self_text", ""))
    # subreddit variants
    df["subreddit"] = df.get("subreddit", df.get("subreddit_name_prefixed", df.get("subreddit", None)))
    df["permalink"] = df.get("permalink", df.get("url", None))
    # Keep any existing timestamp fields if provided by the collector: prefer created_utc -> created_at -> created -> comment_published_at
    df["created_utc"] = df.get("created_utc",
                               df.get("created_at",
                                      df.get("created",
                                             df.get("comment_published_at", None))))
    df["author"] = df.get("author", None)
    df["score"] = df.get("score", df.get("ups", None))
    df["num_comments"] = df.get("num_comments", df.get("num_comments", None))
    return df


def _normalize_reddit_comments(comments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize reddit comments DataFrame.
    Preserve created_utc / created_at / created when present.
    """
    if comments_df is None or comments_df.empty:
        return pd.DataFrame()
    df = comments_df.copy()
    # map comment text
    if "body" in df.columns:
        df["comment_text"] = df["body"]
    elif "text" in df.columns:
        df["comment_text"] = df["text"]
    else:
        df["comment_text"] = df.get("selftext", df.get("comment_text", ""))
    # Preserve timestamps if provided
    df["created_utc"] = df.get("created_utc",
                               df.get("created_at",
                                      df.get("created",
                                             df.get("comment_published_at", None))))
    df["submission_id"] = df.get("submission_id", df.get("link_id", df.get("post_id", None)))
    df["id"] = df.get("id")
    df["parent_id"] = df.get("parent_id")
    df["author"] = df.get("author")
    df["score"] = df.get("score")
    return df


# ---------------- improved reddit fetch ----------------
def fetch_reddit_live_for_keyword(keyword, post_limit=30, comment_limit=100, subs=None, search_mode=False, status_callback=None):
    rc = RedditCollector()
    subs_to_try = subs or getattr(config, "REDDIT_SUBREDDITS", None) or ["all"]
    all_posts = []
    all_comments = []

    kw = str(keyword).strip()
    if not kw:
        return pd.DataFrame(), pd.DataFrame()
    kw_lower = kw.lower()
    kw_variants = {kw_lower, kw_lower.replace(" ", ""), kw_lower.replace(" ", "_")}
    if kw_lower == "elon":
        kw_variants.add("musk")

    for sub in subs_to_try:
        try:
            if status_callback:
                status_callback(f"Fetching posts from r/{sub} (limit={post_limit})...")
            posts_df = pd.DataFrame()
            if search_mode and hasattr(rc, "fetch_posts_by_keyword"):
                try:
                    posts_df = rc.fetch_posts_by_keyword(keyword, limit=post_limit, subreddit=sub)
                except Exception:
                    posts_df = pd.DataFrame()
            if posts_df is None or posts_df.empty:
                try:
                    posts_df = rc.fetch_hot_posts(sub, limit=post_limit)
                except Exception:
                    posts_df = pd.DataFrame()

            if posts_df is None or posts_df.empty:
                continue

            posts_df = _normalize_reddit_posts(posts_df)

            def post_matches_dict(d):
                try:
                    fields = []
                    for f in ("title", "selftext", "permalink", "url", "author", "flair_text", "link_flair_text"):
                        val = d.get(f, "")
                        if val and not isinstance(val, str):
                            val = str(val)
                        fields.append((val or "").lower())
                    joined = " ".join([s for s in fields if s])
                    for v in kw_variants:
                        if v in joined:
                            return True
                    if re.search(r"\b" + re.escape(kw_lower) + r"\b", joined):
                        return True
                except Exception:
                    return False
                return False

            if not search_mode:
                mask = posts_df.apply(lambda r: post_matches_dict(r.to_dict()), axis=1)
                posts_df = posts_df[mask]

            if posts_df.empty:
                continue

            all_posts.append(posts_df)

            top_n_for_comments = min(5, len(posts_df))
            for i in range(top_n_for_comments):
                post_id = str(posts_df.iloc[i]["id"])
                if status_callback:
                    status_callback(f"Fetching comments for post {post_id} (limit={comment_limit})...")
                try:
                    comments_df = rc.fetch_comments_for_post(post_id, limit=comment_limit)
                except Exception:
                    comments_df = pd.DataFrame()
                comments_df = _normalize_reddit_comments(comments_df)
                if comments_df is None or comments_df.empty:
                    continue

                try:
                    bodies = comments_df["comment_text"].astype(str).str.lower().fillna("")
                    match_any = bodies.apply(lambda s: any(v in s for v in kw_variants) or bool(re.search(r"\b" + re.escape(kw_lower) + r"\b", s)))
                except Exception:
                    match_any = pd.Series([False] * len(comments_df))

                comments_df["keyword_in_comment"] = match_any.values if len(match_any) == len(comments_df) else False
                comments_df["post_id"] = post_id
                comments_df["post_title"] = posts_df.iloc[i].get("title", "")
                comments_df["subreddit"] = posts_df.iloc[i].get("subreddit", None)
                all_comments.append(comments_df)
        except Exception as e:
            if status_callback:
                status_callback(f"Error fetching from r/{sub}: {e}")
            continue

    posts_concat = safe_concat_rows(all_posts) if all_posts else pd.DataFrame()
    comments_concat = safe_concat_rows(all_comments) if all_comments else pd.DataFrame()

    if not posts_concat.empty and "id" in posts_concat.columns:
        posts_concat = posts_concat.drop_duplicates(subset=["id"])
    if not comments_concat.empty and "id" in comments_concat.columns:
        comments_concat = comments_concat.drop_duplicates(subset=["id"])

    return posts_concat, comments_concat


# ---------------- load + process combined ----------------
@st.cache_data(ttl=3600)
def load_and_process_data(keyword, video_limit, comment_limit, include_reddit=False,
                          reddit_post_limit=30, reddit_comment_limit=100, reddit_subs=None, reddit_search_mode=False):
    dbh = MongoHandler()

    # YouTube loading (existing)
    try:
        yt_df = dbh.find_data_by_keyword(keyword)
    except Exception as e:
        logging.warning("db find_data_by_keyword failed: %s", e)
        yt_df = pd.DataFrame()
    if yt_df is None:
        yt_df = pd.DataFrame()

    if yt_df.empty:
        try:
            collector = YouTubeCollector()
            fetched = collector.fetch_video_data_by_keyword(keyword, video_limit=video_limit, comment_limit=comment_limit)
            if fetched is not None and not fetched.empty:
                try:
                    dbh.insert_data(fetched)
                except Exception:
                    pass
                try:
                    yt_df = dbh.find_data_by_keyword(keyword)
                except Exception:
                    yt_df = fetched.copy()
        except Exception as e:
            logging.warning("YouTube fetch failed: %s", e)
            yt_df = pd.DataFrame()

    if not yt_df.empty:
        yt_df = ensure_unique_columns(yt_df.copy())
        if "comment_text" not in yt_df.columns:
            if "body" in yt_df.columns:
                yt_df = yt_df.rename(columns={"body": "comment_text"})
            elif "text" in yt_df.columns:
                yt_df = yt_df.rename(columns={"text": "comment_text"})
        yt_df["source"] = "youtube"
        yt_df = _ensure_timestamp_col(yt_df)
        if "cleaned_text" not in yt_df.columns or "sentiment" not in yt_df.columns:
            yt_df["language"] = yt_df["comment_text"].apply(lambda t: detect_language(t) if t else "unknown")
            yt_df["cleaned_text"] = yt_df.apply(lambda r: clean_text_multilingual(r["comment_text"], r["language"]), axis=1)
            yt_df["sentiment"] = yt_df.apply(lambda r: get_sentiment(r["cleaned_text"]) if r["language"] == "en" else "N/A (Non-English)", axis=1)

    combined = yt_df if not yt_df.empty else pd.DataFrame()

    # Reddit live fetch & merge
    if include_reddit:
        try:
            posts_df, comments_df = fetch_reddit_live_for_keyword(
                keyword,
                post_limit=reddit_post_limit,
                comment_limit=reddit_comment_limit,
                subs=reddit_subs,
                search_mode=reddit_search_mode,
                status_callback=lambda s: logging.info("REDDIT_FETCH: %s", s)
            )
        except Exception as e:
            logging.warning("Live reddit fetch failed: %s", e)
            posts_df, comments_df = pd.DataFrame(), pd.DataFrame()

        posts_df = ensure_unique_columns(posts_df) if not posts_df.empty else pd.DataFrame()
        comments_df = ensure_unique_columns(comments_df) if not comments_df.empty else pd.DataFrame()

        # Build reddit_live_df by combining both comments and posts-derived rows
        reddit_comment_rows = pd.DataFrame()
        reddit_post_rows = pd.DataFrame()

        if not comments_df.empty:
            # rename created_utc -> comment_published_at, id -> comment_id
            comments_df = comments_df.rename(columns={"comment_text": "comment_text", "created_utc": "comment_published_at", "id": "comment_id"})
            comments_df["source"] = "reddit"
            comments_df["origin"] = "reddit_live_fetch"
            comments_df["channel_title"] = comments_df.get("subreddit", None)
            comments_df["language"] = comments_df["comment_text"].apply(lambda t: detect_language(t) if t else "unknown")
            comments_df["cleaned_text"] = comments_df.apply(lambda r: clean_text_multilingual(r["comment_text"], r["language"]), axis=1)
            comments_df["sentiment"] = comments_df.apply(lambda r: get_sentiment(r["cleaned_text"]) if r["language"] == "en" else "N/A (Non-English)", axis=1)
            # Use robust timestamp detection
            comments_df = _ensure_timestamp_col(comments_df)
            reddit_comment_rows = comments_df

        if not posts_df.empty:
            temp = posts_df.copy()
            temp["comment_text"] = temp.get("selftext", "").fillna("") + " " + temp.get("title", "").fillna("")
            # Prefer created_utc then created_at then created then comment_published_at
            if "created_utc" in temp.columns and temp["created_utc"].notna().any():
                temp["comment_published_at"] = temp["created_utc"]
            elif "created_at" in temp.columns and temp["created_at"].notna().any():
                temp["comment_published_at"] = temp["created_at"]
            elif "created" in temp.columns and temp["created"].notna().any():
                temp["comment_published_at"] = temp["created"]
            else:
                # fallback: try to detect any time-like column using our helper
                best = _find_best_time_column(temp)
                if best:
                    temp["comment_published_at"] = temp[best]
                else:
                    temp["comment_published_at"] = None
            temp["language"] = temp["comment_text"].apply(lambda t: detect_language(t) if t else "unknown")
            temp["cleaned_text"] = temp.apply(lambda r: clean_text_multilingual(r["comment_text"], r["language"]), axis=1)
            temp["sentiment"] = temp.apply(lambda r: get_sentiment(r["cleaned_text"]) if r["language"] == "en" else "N/A (Non-English)", axis=1)
            temp["source"] = "reddit"
            temp["origin"] = "reddit_live_posts"
            temp["channel_title"] = temp.get("subreddit", None)
            # create a pseudo comment_id for posts to satisfy db_handler expectations and to allow dedupe
            if "comment_id" not in temp.columns:
                temp["comment_id"] = None
            # ensure proper timestamp conversion
            temp = _ensure_timestamp_col(temp)
            reddit_post_rows = temp

        # Now unify comments + posts rows for analysis
        reddit_live_df = pd.DataFrame()
        if not reddit_comment_rows.empty and not reddit_post_rows.empty:
            reddit_live_df = safe_concat_rows([reddit_comment_rows, reddit_post_rows])
        elif not reddit_comment_rows.empty:
            reddit_live_df = reddit_comment_rows
        elif not reddit_post_rows.empty:
            reddit_live_df = reddit_post_rows

        # Insert freshly fetched reddit posts and comments into DB (dedupe where possible)
        try:
            collection = getattr(dbh, "collection", None)
            # Insert posts
            if not posts_df.empty:
                new_posts = posts_df.copy()
                # ensure comment_id exists to avoid insert_data KeyError in previous runs
                if "comment_id" not in new_posts.columns:
                    new_posts["comment_id"] = None
                # dedupe by id in DB if collection available
                if collection is not None and "id" in new_posts.columns:
                    ids = [str(x) for x in new_posts["id"].dropna().unique().tolist()]
                    existing = set()
                    if ids:
                        for d in collection.find({"id": {"$in": ids}}, {"id": 1}):
                            existing.add(str(d.get("id")))
                    new_posts = new_posts[~new_posts["id"].astype(str).isin(existing)].copy()
                if not new_posts.empty:
                    try:
                        dbh.insert_data(new_posts)
                        logging.info("Inserted/Updated %d reddit posts via db_handler.insert_data().", len(new_posts))
                    except Exception as e:
                        logging.warning("High-level insert posts failed, trying raw insert: %s", e)
                        try:
                            if collection is not None:
                                collection.insert_many(new_posts.to_dict(orient="records"))
                                logging.info("Raw inserted %d reddit posts.", len(new_posts))
                            else:
                                dbh.insert_data(new_posts)
                                logging.info("Inserted (fallback) %d reddit posts.", len(new_posts))
                        except Exception as e2:
                            logging.error("Raw insert posts also failed: %s", e2)

            # Insert comments
            if not comments_df.empty:
                new_comments = comments_df.copy()
                if "comment_id" not in new_comments.columns and "id" in new_comments.columns:
                    new_comments = new_comments.rename(columns={"id": "comment_id"})
                if "comment_id" not in new_comments.columns:
                    new_comments["comment_id"] = None
                if collection is not None and "comment_id" in new_comments.columns:
                    cids = [str(x) for x in new_comments["comment_id"].dropna().unique().tolist()]
                    existing_c = set()
                    if cids:
                        for d in collection.find({"comment_id": {"$in": cids}}, {"comment_id": 1}):
                            existing_c.add(str(d.get("comment_id")))
                    if cids:
                        new_comments = new_comments[~new_comments["comment_id"].astype(str).isin(existing_c)].copy()
                if not new_comments.empty:
                    try:
                        dbh.insert_data(new_comments)
                        logging.info("Inserted/Updated %d reddit comments via db_handler.insert_data().", len(new_comments))
                    except Exception as e:
                        logging.warning("High-level insert comments failed, trying raw insert. Error: %s", e)
                        try:
                            if collection is not None:
                                collection.insert_many(new_comments.to_dict(orient="records"))
                                logging.info("Raw inserted %d reddit comments.", len(new_comments))
                            else:
                                dbh.insert_data(new_comments)
                                logging.info("Inserted (fallback) %d reddit comments.", len(new_comments))
                        except Exception as e2:
                            logging.error("Raw insert comments also failed: %s", e2)
        except Exception as e:
            logging.error("Inserting fetched reddit data into DB failed: %s", e)

        # Merge reddit_live_df into combined safely
        try:
            if reddit_live_df is not None and not reddit_live_df.empty:
                combined = safe_concat_rows([combined, reddit_live_df]) if (combined is not None and not combined.empty) else reddit_live_df.copy()
        except Exception as e:
            logging.error("Merge (concat) of reddit live into combined failed: %s", e)
            try:
                if reddit_live_df is not None and not reddit_live_df.empty and combined is not None and not combined.empty:
                    common = [c for c in combined.columns if c in reddit_live_df.columns]
                    if common:
                        combined = pd.concat([combined[common], reddit_live_df[common]], ignore_index=True, sort=False)
                    else:
                        combined = safe_concat_rows([combined, reddit_live_df])
            except Exception as e2:
                logging.error("Fallback merge also failed: %s", e2)

    if combined is None or (isinstance(combined, pd.DataFrame) and combined.empty):
        return None

    # final robust timestamp handling for the merged combined DF
    combined = _ensure_timestamp_col(combined)

    if "cleaned_text" not in combined.columns:
        combined["language"] = combined["comment_text"].apply(lambda t: detect_language(t) if t else "unknown")
        combined["cleaned_text"] = combined.apply(lambda r: clean_text_multilingual(r["comment_text"], r["language"]), axis=1)
        combined["sentiment"] = combined.apply(lambda r: get_sentiment(r["cleaned_text"]) if r["language"] == "en" else "N/A (Non-English)", axis=1)

    return combined


# -------------------- UI --------------------
st.title("📺 TrendPulse AI: Multilingual YouTube + Reddit Analytics")

with st.sidebar:
    st.header("⚙️ Controls")
    # Prefilled keyword set to "terrorist" as requested
    keyword_input = st.text_input("Enter Search Keyword", value="terrorist")
    # YouTube defaults as per screenshot
    video_limit = st.slider("Videos to Scan (YouTube)", 5, 50, 10)
    comment_limit = st.slider("Comments per Video (YouTube)", 10, 100, 20)
    analyze_button = st.button("Analyze Trends", type="primary")

with st.sidebar:
    st.markdown("---")
    st.subheader("🔗 Reddit Controls (live fetch)")
    # Force default UI to show Reddit enabled by default
    default_enabled = True
    default_post_limit = 500
    default_comment_limit = 100

    reddit_enabled_ui = st.checkbox("Enable Reddit (override config)", value=default_enabled)
    reddit_post_limit = st.number_input("Reddit posts to fetch per subreddit/search", min_value=1, max_value=2000, value=default_post_limit, step=1)
    reddit_comment_limit = st.number_input("Comments per reddit post to fetch", min_value=1, max_value=1000, value=default_comment_limit, step=1)

    # reddit_mode is fixed to Search r/all for keyword; no configured-subreddits option
    reddit_subs = None

    st.markdown("---")
    st.markdown("Manual actions")
    fetch_now = st.button("Fetch Reddit Now (manual)")

# Analyze action
if analyze_button and keyword_input:
    with st.spinner(f"Analyzing '{keyword_input}'..."):
        try:
            df = load_and_process_data(
                keyword_input,
                video_limit,
                comment_limit,
                include_reddit=reddit_enabled_ui,
                reddit_post_limit=int(reddit_post_limit),
                reddit_comment_limit=int(reddit_comment_limit),
                reddit_subs=reddit_subs,
                reddit_search_mode=True  # always search r/all for keyword
            )
        except Exception as e:
            st.error(f"Data loading failed: {e}")
            st.exception(traceback.format_exc())
            df = None

    if df is not None and not df.empty:
        st.success(f"Analysis complete! Processed {len(df)} rows (YouTube + Reddit live).")
        source_counts = df["source"].value_counts().to_dict() if "source" in df.columns else {"youtube": len(df)}
        st.markdown(f"**Data sources present:** {source_counts}")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏆 Top Content", "📺 Top Channels", "📝 Raw Data"])

        # Overview
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sentiment (English only) — Combined")
                sentiment_counts = df[df["sentiment"] != "N/A (Non-English)"]["sentiment"].value_counts()
                if not sentiment_counts.empty:
                    fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, title="Combined Sentiment")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No English comments found to analyze sentiment.")
                st.subheader("Most Frequent Words (Combined)")
                text = " ".join(df["cleaned_text"].astype(str))
                if text.strip():
                    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc, interpolation="bilinear")
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)
            with col2:
                st.subheader("Sentiment by Source")
                y_counts = df[df["source"] == "youtube"]["sentiment"].value_counts()
                r_counts = df[df["source"] == "reddit"]["sentiment"].value_counts() if "reddit" in df["source"].values else pd.Series()
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**YouTube**")
                    if not y_counts.empty:
                        fig_y = px.pie(y_counts, values=y_counts.values, names=y_counts.index, title="YouTube Sentiment")
                        st.plotly_chart(fig_y, use_container_width=True)
                    else:
                        st.info("No English YouTube comments to show.")
                with c2:
                    st.markdown("**Reddit**")
                    if not r_counts.empty:
                        fig_r = px.pie(r_counts, values=r_counts.values, names=r_counts.index, title="Reddit Sentiment")
                        st.plotly_chart(fig_r, use_container_width=True)
                    else:
                        st.info("No English Reddit comments to show.")

            # -------------------- Sentiment Heatmap (Combined) --------------------
            st.subheader("📌 Sentiment Heatmap — YouTube + Reddit")
            try:
                # Filter out non-English (because sentiment for them is N/A)
                df_sent = df[df["sentiment"] != "N/A (Non-English)"].copy()

                if df_sent.empty:
                    st.info("No English comments for heatmap.")
                else:
                    # Create pivot for heatmap
                    pivot_df = (
                        df_sent.pivot_table(
                            index="sentiment",
                            columns="source",
                            values="comment_text",
                            aggfunc="count",
                            fill_value=0
                        )
                        .reset_index()
                    )

                    # Convert pivoted table to long format for Plotly heatmap
                    heatmap_data = pivot_df.set_index("sentiment")

                    fig_heatmap = px.imshow(
                        heatmap_data,
                        text_auto=True,
                        aspect="auto",
                        labels=dict(x="Source", y="Sentiment", color="Count"),
                        title="Sentiment Distribution Heatmap (YouTube + Reddit)"
                    )

                    st.plotly_chart(fig_heatmap, use_container_width=True)

            except Exception as e:
                st.error(f"Could not render sentiment heatmap: {e}")

            # ----- Sentiment Trend Over Time -----
            st.subheader("Sentiment Trend Over Time (English Only)")
            try:
                df_sent = df[df["sentiment"] != "N/A (Non-English)"].copy()
                df_sent = _ensure_timestamp_col(df_sent)
                # If date column conversion fails, guard it
                if "comment_published_at" in df_sent.columns:
                    df_sent["date"] = df_sent["comment_published_at"].dt.date
                else:
                    df_sent["date"] = pd.NaT

                sent_timeline = (
                    df_sent.groupby(["date", "sentiment"])
                    .size()
                    .reset_index(name="count")
                )

                if not sent_timeline.empty:
                    fig_sent_time = px.line(
                        sent_timeline,
                        x="date",
                        y="count",
                        color="sentiment",
                        markers=True,
                        title="Sentiment Trend Over Time"
                    )
                    st.plotly_chart(fig_sent_time, use_container_width=True)
                else:
                    st.info("No sentiment timeline data available.")
            except Exception as e:
                st.info("Could not plot sentiment trend over time: " + str(e))

            # ----- Top Keywords (word frequency) -----
            st.subheader("Top Keywords (Word Frequency)")
            try:
                from collections import Counter

                words = " ".join(df["cleaned_text"].astype(str)).split()
                freq = Counter(words).most_common(20)
                freq_df = pd.DataFrame(freq, columns=["word", "count"])

                fig_kw_bar = px.bar(
                    freq_df,
                    x="word",
                    y="count",
                    title="Top 20 Words in Cleaned Text",
                    text="count"
                )
                st.plotly_chart(fig_kw_bar, use_container_width=True)

            except Exception as e:
                st.info("Could not compute top keywords: " + str(e))

        # Trends & Top Content
        with tab2:
            st.subheader("Comment Activity Over Time (Combined & Per Source)")
            try:
                df_tr = _ensure_timestamp_col(df.copy())
                df_tr_nonull = df_tr.dropna(subset=["comment_published_at"])
                if df_tr_nonull.empty:
                    st.info("Combined trend not available: no valid timestamps in combined data.")
                else:
                    combined_trends = get_trends_over_time(df_tr_nonull)
                    if combined_trends is not None and not combined_trends.empty:
                        fig_comb = px.line(combined_trends, x="comment_published_at", y="count", title="Combined Comment Volume per Day")
                        st.plotly_chart(fig_comb, use_container_width=True)
                    else:
                        st.info("Combined trend calculation returned no rows.")
            except Exception as e:
                st.info("Combined trends unavailable: " + str(e))

            col_y, col_r = st.columns(2)
            with col_y:
                st.markdown("**YouTube Comment Volume**")
                try:
                    ydf = df[df["source"] == "youtube"].copy()
                    ydf = _ensure_timestamp_col(ydf)
                    ydf_nonull = ydf.dropna(subset=["comment_published_at"])
                    if ydf_nonull.empty:
                        st.info("No YouTube timestamps available to plot trends.")
                    else:
                        y_tr = get_trends_over_time(ydf_nonull)
                        if y_tr is not None and not y_tr.empty:
                            fig_y = px.line(y_tr, x="comment_published_at", y="count", title="YouTube")
                            st.plotly_chart(fig_y, use_container_width=True)
                        else:
                            st.info("No YouTube trend data available.")
                except Exception as e:
                    st.info("YouTube trends failed: " + str(e))

            with col_r:
                st.markdown("**Reddit Comment Volume**")
                try:
                    rdf = df[df["source"] == "reddit"].copy()
                    rdf = _ensure_timestamp_col(rdf)
                    rdf_nonull = rdf.dropna(subset=["comment_published_at"])
                    if rdf_nonull.empty:
                        st.info("No Reddit timestamps available to plot trends.")
                    else:
                        r_tr = get_trends_over_time(rdf_nonull)
                        if r_tr is not None and not r_tr.empty:
                            fig_r = px.line(r_tr, x="comment_published_at", y="count", title="Reddit")
                            st.plotly_chart(fig_r, use_container_width=True)
                        else:
                            st.info("No Reddit trend data available.")
                except Exception as e:
                    st.info("Reddit trends failed: " + str(e))

            # ---------- New: Side-by-side small line graphs above Top Content ----------
            st.markdown("---")
            st.subheader("Small Trend Comparison (Recent)")
            try:
                # Use last N days window (e.g., 30 days) for small charts
                days_window = 30
                # prepare combined daily
                df_days = _ensure_timestamp_col(df.copy()).dropna(subset=["comment_published_at"]).copy()
                if not df_days.empty:
                    df_days["date"] = df_days["comment_published_at"].dt.date
                    comb_daily = df_days.groupby("date").size().reset_index(name="count")
                else:
                    comb_daily = pd.DataFrame(columns=["date", "count"])

                # youtube daily
                ydf_days = df[df["source"] == "youtube"].copy()
                ydf_days = _ensure_timestamp_col(ydf_days).dropna(subset=["comment_published_at"]).copy()
                if not ydf_days.empty:
                    ydf_days["date"] = ydf_days["comment_published_at"].dt.date
                    y_daily = ydf_days.groupby("date").size().reset_index(name="count")
                else:
                    y_daily = pd.DataFrame(columns=["date", "count"])

                # reddit daily
                rdf_days = df[df["source"] == "reddit"].copy()
                rdf_days = _ensure_timestamp_col(rdf_days).dropna(subset=["comment_published_at"]).copy()
                if not rdf_days.empty:
                    rdf_days["date"] = rdf_days["comment_published_at"].dt.date
                    r_daily = rdf_days.groupby("date").size().reset_index(name="count")
                else:
                    r_daily = pd.DataFrame(columns=["date", "count"])

                # restrict to recent window if possible
                if not comb_daily.empty:
                    max_date = comb_daily["date"].max()
                elif not y_daily.empty:
                    max_date = y_daily["date"].max()
                elif not r_daily.empty:
                    max_date = r_daily["date"].max()
                else:
                    max_date = None

                if max_date is not None:
                    import datetime as _dt
                    min_date = max_date - _dt.timedelta(days=days_window)
                    if not comb_daily.empty:
                        comb_daily = comb_daily[comb_daily["date"] >= min_date]
                    if not y_daily.empty:
                        y_daily = y_daily[y_daily["date"] >= min_date]
                    if not r_daily.empty:
                        r_daily = r_daily[r_daily["date"] >= min_date]

                scol1, scol2, scol3 = st.columns(3)
                with scol1:
                    st.markdown("**Combined (last 30 days)**")
                    if not comb_daily.empty:
                        fig_c_small = px.line(comb_daily, x="date", y="count", title="", markers=True)
                        fig_c_small.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=250)
                        st.plotly_chart(fig_c_small, use_container_width=True)
                    else:
                        st.info("No combined data for small chart.")
                with scol2:
                    st.markdown("**YouTube (last 30 days)**")
                    if not y_daily.empty:
                        fig_y_small = px.line(y_daily, x="date", y="count", title="", markers=True)
                        fig_y_small.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=250)
                        st.plotly_chart(fig_y_small, use_container_width=True)
                    else:
                        st.info("No YouTube data for small chart.")
                with scol3:
                    st.markdown("**Reddit (last 30 days)**")
                    if not r_daily.empty:
                        fig_r_small = px.line(r_daily, x="date", y="count", title="", markers=True)
                        fig_r_small.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=250)
                        st.plotly_chart(fig_r_small, use_container_width=True)
                    else:
                        st.info("No Reddit data for small chart.")
            except Exception as e:
                st.info("Could not generate small trend comparison: " + str(e))

            st.subheader("Top Content (YouTube videos + Reddit posts)")
            try:
                posts = []
                ydf = df[df["source"] == "youtube"]
                if not ydf.empty:
                    key = "video_title" if "video_title" in ydf.columns else "title"
                    if key in ydf.columns:
                        grp = ydf.groupby(key).agg(comments=("comment_text", "count")).reset_index()
                        for _, r in grp.sort_values("comments", ascending=False).head(30).iterrows():
                            posts.append({"title": r[key], "source": "youtube", "comment_count": int(r["comments"])})
                rdf = df[df["source"] == "reddit"]
                if not rdf.empty:
                    key_r = "post_title" if "post_title" in rdf.columns else "title"
                    if key_r in rdf.columns:
                        grp2 = rdf.groupby(key_r).agg(comments=("comment_text", "count")).reset_index()
                        for _, r in grp2.sort_values("comments", ascending=False).head(30).iterrows():
                            posts.append({"title": r[key_r], "source": "reddit", "comment_count": int(r["comments"])})
                posts_df = pd.DataFrame(posts).sort_values("comment_count", ascending=False).head(30)
                st.dataframe(posts_df.fillna(""))
            except Exception as e:
                st.info("Could not compute combined top content: " + str(e))

            # ----- Keyword Mention Timeline -----
            st.subheader("Keyword Mentions Over Time")
            try:
                df_mentions = df.copy()
                df_mentions = _ensure_timestamp_col(df_mentions)
                df_mentions = df_mentions.dropna(subset=["comment_published_at"])

                # Count rows (comments/posts) per day per source
                timeline = (
                    df_mentions.groupby([df_mentions["comment_published_at"].dt.date, "source"])
                    .size()
                    .reset_index(name="mentions")
                    .rename(columns={"comment_published_at": "date"})
                )

                fig_kw = px.line(
                    timeline,
                    x="date",
                    y="mentions",
                    color="source",
                    markers=True,
                    title="Daily Keyword Mentions (YouTube vs Reddit)"
                )
                st.plotly_chart(fig_kw, use_container_width=True)

            except Exception as e:
                st.info("Keyword mention timeline could not be generated: " + str(e))

        # Top channels & subreddits
        with tab3:
            st.subheader("Top Channels & Subreddits")
            rows = []
            try:
                ydf = df[df["source"] == "youtube"]
                if not ydf.empty and "channel_title" in ydf.columns:
                    ch_stats = ydf.groupby("channel_title").agg(videos=("source", "size"), comments=("comment_text", "count")).reset_index()
                    for _, r in ch_stats.sort_values("comments", ascending=False).head(30).iterrows():
                        rows.append({"name": r["channel_title"], "type": "youtube_channel", "videos_or_posts": int(r["videos"]), "total_comments": int(r["comments"])})
                rdf = df[df["source"] == "reddit"]
                if not rdf.empty and "subreddit" in rdf.columns:
                    submission_col = "submission_id" if "submission_id" in rdf.columns else ("source_id" if "source_id" in rdf.columns else None)
                    if submission_col:
                        sr_stats = rdf.groupby("subreddit").agg(posts=(submission_col, "nunique"), comments=("comment_text", "count")).reset_index()
                    else:
                        sr_stats = rdf.groupby("subreddit").agg(comments=("comment_text", "count")).reset_index()
                        sr_stats["posts"] = 0
                    for _, r in sr_stats.sort_values("comments", ascending=False).head(30).iterrows():
                        rows.append({"name": r["subreddit"], "type": "reddit_subreddit", "videos_or_posts": int(r.get("posts", 0)) if pd.notna(r.get("posts", 0)) else 0, "total_comments": int(r["comments"])})
            except Exception as e:
                logging.warning("Top channels calc failed: %s", e)
            ch_df = pd.DataFrame(rows).sort_values("total_comments", ascending=False).head(50)
            if not ch_df.empty:
                st.dataframe(ch_df)
            else:
                st.info("No channel/subreddit metrics available.")
            # ----- Engagement Distribution -----
            st.subheader("View Count vs Like Count (YouTube Video Engagement)")

            try:
                ydf = df[df["source"] == "youtube"]
                if not ydf.empty and "view_count" in ydf.columns and "like_count" in ydf.columns:
                    fig_eng = px.scatter(
                        ydf,
                        x="view_count",
                        y="like_count",
                        color="channel_title",
                        hover_data=["video_title"],
                        title="Engagement Distribution: Views vs Likes",
                    )
                    st.plotly_chart(fig_eng, use_container_width=True)
                else:
                    st.info("YouTube data not sufficient for engagement plot.")

            except Exception as e:
                st.info("Could not generate engagement scatter: " + str(e))

            # ----- Channel Performance Radar Chart -----
            st.subheader("Channel Performance Radar (Top 5 Channels)")

            try:
                from src.analysis.influencers import get_top_channels
                top_ch = get_top_channels(df[df["source"]=="youtube"], top_n=5)

                if not top_ch.empty:
                    radar_df = top_ch.melt(id_vars="channel_title",
                                        value_vars=["total_views", "total_likes", "comments_in_sample", "engagement_score"],
                                        var_name="metric",
                                        value_name="value")

                    fig_radar = px.line_polar(
                        radar_df,
                        r="value",
                        theta="metric",
                        color="channel_title",
                        line_close=True,
                        title="Top 5 Channels – Performance Radar"
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("Channel performance data not available.")

            except Exception as e:
                st.info("Could not generate radar chart: " + str(e))
        # Raw data
        with tab4:
            st.subheader("Processed Comment Data (YouTube + Reddit live)")
            display_cols = ["comment_published_at", "comment_text", "cleaned_text", "sentiment", "source", "channel_title", "subreddit", "post_title", "origin"]
            available_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available_cols].head(1000))
    else:
        st.error(f"No data found for '{keyword_input}'. Try increasing Reddit post/comment limits or enabling search mode.")
else:
    st.info("Enter a keyword and click Analyze Trends to fetch YouTube and (if enabled) Reddit live data.")


# Manual fetch button
if fetch_now:
    status = st.empty()
    try:
        if not reddit_enabled_ui:
            status.info("Reddit is disabled in the sidebar. Enable it to fetch.")
        else:
            try:
                rc = RedditCollector()
            except Exception as e:
                status.error(f"Could not init RedditCollector: {e}")
                rc = None
            try:
                dbh = MongoHandler()
            except Exception as e:
                status.error(f"Could not init MongoHandler: {e}")
                dbh = None

            if rc is None or dbh is None:
                status.error("Collector or DB handler not available.")
            else:
                status.info("Fetching Reddit (manual)...")
                posts_df, comments_df = fetch_reddit_live_for_keyword(
                    keyword_input,
                    post_limit=int(reddit_post_limit),
                    comment_limit=int(reddit_comment_limit),
                    # subs=reddit_subs,
                    search_mode=True,  # always search r/all for keyword
                    status_callback=lambda s: status.text(s)
                )
                try:
                    collection = getattr(dbh, "collection", None)
                    if posts_df is not None and not posts_df.empty:
                        new_posts = posts_df.copy()
                        if "comment_id" not in new_posts.columns:
                            new_posts["comment_id"] = None
                        if collection is not None and "id" in new_posts.columns:
                            ids = [str(x) for x in new_posts["id"].dropna().unique().tolist()]
                            existing = set()
                            if ids:
                                for d in collection.find({"id": {"$in": ids}}, {"id": 1}):
                                    existing.add(str(d.get("id")))
                            new_posts = new_posts[~new_posts["id"].astype(str).isin(existing)].copy()
                        if not new_posts.empty:
                            dbh.insert_data(new_posts)
                            status.success(f"Inserted/Updated {len(new_posts)} posts.")
                    if comments_df is not None and not comments_df.empty:
                        new_comments = comments_df.copy()
                        if "comment_id" not in new_comments.columns and "id" in new_comments.columns:
                            new_comments = new_comments.rename(columns={"id": "comment_id"})
                        if "comment_id" not in new_comments.columns:
                            new_comments["comment_id"] = None
                        if collection is not None and "comment_id" in new_comments.columns:
                            cids = [str(x) for x in new_comments["comment_id"].dropna().unique().tolist()]
                            existing_c = set()
                            if cids:
                                for d in collection.find({"comment_id": {"$in": cids}}, {"comment_id": 1}):
                                    existing_c.add(str(d.get("comment_id")))
                            new_comments = new_comments[~new_comments["comment_id"].astype(str).isin(existing_c)].copy()
                        if not new_comments.empty:
                            dbh.insert_data(new_comments)
                            status.success(f"Inserted/Updated {len(new_comments)} comments.")
                except Exception as e:
                    status.error("Insertion failed: " + str(e))
    except Exception as e:
        st.error("Manual fetch failed: " + str(e))
        st.exception(traceback.format_exc())
