# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langdetect import detect, LangDetectException
import traceback
import re

# Project modules
from src.data_collection.collector import YouTubeCollector
from src.data_collection.reddit_collector import RedditCollector
from src.database.db_handler import MongoHandler
from src.processing.preprocessor import clean_text_multilingual
from src.analysis.sentiment import get_sentiment
from src.analysis.trends import get_trends_over_time
import config

st.set_page_config(page_title="TrendPulse AI: YouTube + Reddit", layout="wide", page_icon="📺")


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


def _ensure_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    candidates = [
        "comment_published_at", "publishedAt", "published_at", "created_utc", "created_at",
        "created", "comment_published", "comment_publishedAt", "utc_datetime"
    ]
    found = None
    for c in candidates:
        if c in df.columns:
            found = c
            break
    if found:
        try:
            df["comment_published_at"] = pd.to_datetime(df[found], errors="coerce")
        except Exception:
            df["comment_published_at"] = pd.to_datetime(df[found].astype(str), errors="coerce")
    else:
        df["comment_published_at"] = pd.to_datetime(pd.Series([pd.NaT] * len(df)))
    return df


# ---------------- reddit normalization ----------------
def _normalize_reddit_posts(posts_df: pd.DataFrame) -> pd.DataFrame:
    if posts_df is None or posts_df.empty:
        return pd.DataFrame()
    df = posts_df.copy()
    df["id"] = df.get("id")
    df["title"] = df.get("title", "")
    df["selftext"] = df.get("selftext", df.get("self_text", ""))
    df["subreddit"] = df.get("subreddit", df.get("subreddit_name_prefixed", df.get("subreddit", None)))
    df["permalink"] = df.get("permalink", df.get("url", None))
    df["created_utc"] = df.get("comment_published_at", df.get("created_at", df.get("created", None)))
    df["author"] = df.get("author", None)
    df["score"] = df.get("score", df.get("ups", None))
    df["num_comments"] = df.get("num_comments", df.get("num_comments", None))
    return df


def _normalize_reddit_comments(comments_df: pd.DataFrame) -> pd.DataFrame:
    if comments_df is None or comments_df.empty:
        return pd.DataFrame()
    df = comments_df.copy()
    if "body" in df.columns:
        df["comment_text"] = df["body"]
    elif "text" in df.columns:
        df["comment_text"] = df["text"]
    else:
        df["comment_text"] = df.get("selftext", df.get("comment_text", ""))
    df["created_utc"] = df.get("comment_published_at", df.get("created_at", df.get("created", None)))
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
        print("db find_data_by_keyword failed:", e)
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
            print("YouTube fetch failed:", e)
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
                status_callback=lambda s: print("REDDIT_FETCH:", s)
            )
        except Exception as e:
            print("Live reddit fetch failed:", e)
            posts_df, comments_df = pd.DataFrame(), pd.DataFrame()

        posts_df = ensure_unique_columns(posts_df) if not posts_df.empty else pd.DataFrame()
        comments_df = ensure_unique_columns(comments_df) if not comments_df.empty else pd.DataFrame()

        # Build reddit_live_df by combining both comments and posts-derived rows
        reddit_comment_rows = pd.DataFrame()
        reddit_post_rows = pd.DataFrame()

        if not comments_df.empty:
            comments_df = comments_df.rename(columns={"comment_text": "comment_text", "created_utc": "comment_published_at", "id": "comment_id"})
            comments_df["source"] = "reddit"
            comments_df["origin"] = "reddit_live_fetch"
            comments_df["channel_title"] = comments_df.get("subreddit", None)
            comments_df["language"] = comments_df["comment_text"].apply(lambda t: detect_language(t) if t else "unknown")
            comments_df["cleaned_text"] = comments_df.apply(lambda r: clean_text_multilingual(r["comment_text"], r["language"]), axis=1)
            comments_df["sentiment"] = comments_df.apply(lambda r: get_sentiment(r["cleaned_text"]) if r["language"] == "en" else "N/A (Non-English)", axis=1)
            comments_df = _ensure_timestamp_col(comments_df)
            reddit_comment_rows = comments_df

        if not posts_df.empty:
            temp = posts_df.copy()
            temp["comment_text"] = temp.get("selftext", "").fillna("") + " " + temp.get("title", "").fillna("")
            temp["comment_published_at"] = temp.get("comment_published_at")
            temp["language"] = temp["comment_text"].apply(lambda t: detect_language(t) if t else "unknown")
            temp["cleaned_text"] = temp.apply(lambda r: clean_text_multilingual(r["comment_text"], r["language"]), axis=1)
            temp["sentiment"] = temp.apply(lambda r: get_sentiment(r["cleaned_text"]) if r["language"] == "en" else "N/A (Non-English)", axis=1)
            temp["source"] = "reddit"
            temp["origin"] = "reddit_live_posts"
            temp["channel_title"] = temp.get("subreddit", None)
            # create a pseudo comment_id for posts to satisfy db_handler expectations and to allow dedupe
            if "comment_id" not in temp.columns:
                temp["comment_id"] = None
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
                        print(f"Inserted/Updated {len(new_posts)} reddit posts via db_handler.insert_data().")
                    except Exception as e:
                        print("High-level insert posts failed, trying raw insert:", e)
                        try:
                            if collection is not None:
                                collection.insert_many(new_posts.to_dict(orient="records"))
                                print(f"Raw inserted {len(new_posts)} reddit posts.")
                            else:
                                # fallback: attempt dbh.insert_data with dataframe again
                                dbh.insert_data(new_posts)
                                print(f"Inserted (fallback) {len(new_posts)} reddit posts.")
                        except Exception as e2:
                            print("Raw insert posts also failed:", e2)

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
                        print(f"Inserted/Updated {len(new_comments)} reddit comments via db_handler.insert_data().")
                    except Exception as e:
                        print("High-level insert comments failed, trying raw insert:", e)
                        try:
                            if collection is not None:
                                collection.insert_many(new_comments.to_dict(orient="records"))
                                print(f"Raw inserted {len(new_comments)} reddit comments.")
                            else:
                                dbh.insert_data(new_comments)
                                print(f"Inserted (fallback) {len(new_comments)} reddit comments.")
                        except Exception as e2:
                            print("Raw insert comments also failed:", e2)
        except Exception as e:
            print("Inserting fetched reddit data into DB failed:", e)

        # Merge reddit_live_df into combined safely
        try:
            if reddit_live_df is not None and not reddit_live_df.empty:
                combined = safe_concat_rows([combined, reddit_live_df]) if (combined is not None and not combined.empty) else reddit_live_df.copy()
        except Exception as e:
            print("Merge (concat) of reddit live into combined failed:", e)
            try:
                if reddit_live_df is not None and not reddit_live_df.empty and combined is not None and not combined.empty:
                    common = [c for c in combined.columns if c in reddit_live_df.columns]
                    if common:
                        combined = pd.concat([combined[common], reddit_live_df[common]], ignore_index=True, sort=False)
                    else:
                        combined = safe_concat_rows([combined, reddit_live_df])
            except Exception as e2:
                print("Fallback merge also failed:", e2)

    if combined is None or (isinstance(combined, pd.DataFrame) and combined.empty):
        return None

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
    keyword_input = st.text_input("Enter Search Keyword", value="Nvidia")
    video_limit = st.slider("Videos to Scan (YouTube)", 5, 50, 10)
    comment_limit = st.slider("Comments per Video (YouTube)", 10, 100, 20)
    analyze_button = st.button("Analyze Trends", type="primary")

with st.sidebar:
    st.markdown("---")
    st.subheader("🔗 Reddit Controls (live fetch)")
    default_enabled = getattr(config, "REDDIT_ENABLED", False)
    default_post_limit = getattr(config, "REDDIT_POST_LIMIT", 30)
    # default_subs = getattr(config, "REDDIT_SUBREDDITS", ["technology"])

    reddit_enabled_ui = st.checkbox("Enable Reddit (override config)", value=default_enabled)
    reddit_post_limit = st.number_input("Reddit posts to fetch per subreddit/search", min_value=1, max_value=500, value=default_post_limit, step=1)
    reddit_comment_limit = st.number_input("Comments per reddit post to fetch", min_value=1, max_value=500, value=100, step=1)
    reddit_mode = st.radio("Reddit fetch mode", ("Search r/all for keyword"))
    # if reddit_mode == "Configured subreddits":
    #     subs_input = st.text_area("Subreddits (comma-separated)", value=",".join(default_subs))
    #     reddit_subs = [s.strip() for s in subs_input.split(",") if s.strip()]
    # else:
    #     reddit_subs = None

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
                # reddit_subs=reddit_subs,
                reddit_search_mode=(reddit_mode == "Search r/all for keyword")
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
                    rdf_nonull = rdf.dropna(subset=["created_utc", "comment_published_at"])
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
                print("Top channels calc failed:", e)
            ch_df = pd.DataFrame(rows).sort_values("total_comments", ascending=False).head(50)
            if not ch_df.empty:
                st.dataframe(ch_df)
            else:
                st.info("No channel/subreddit metrics available.")

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
                    search_mode=(reddit_mode == "Search r/all for keyword"),
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
