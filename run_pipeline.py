import os
from src.data_collection.collector import YouTubeCollector
from src.data_collection.reddit_collector import RedditCollector
from src.database.db_handler import MongoHandler
import config


def _cfg_env_or_config(name, default=None):
    """Prefer environment variables over config.py values."""
    return os.getenv(name, getattr(config, name, default))


def _instantiate_mongo_handler(mongo_uri, mongo_db_name):
    """
    Try to instantiate MongoHandler with several possible constructor signatures:
    1) MongoHandler(mongo_uri, mongo_db_name)
    2) MongoHandler(mongo_uri)
    3) MongoHandler()
    Return the instance or raise the last exception.
    """
    last_exc = None
    try:
        return MongoHandler(mongo_uri, mongo_db_name)
    except Exception as e:
        last_exc = e
    try:
        return MongoHandler(mongo_uri)
    except Exception as e:
        last_exc = e
    try:
        return MongoHandler()
    except Exception as e:
        last_exc = e
    raise last_exc


def run_daily_ingestion():
    """
    This function runs the daily data ingestion pipeline.
    It fetches data for YouTube and Reddit (for the same keywords), then stores results in MongoDB.
    """
    keywords_to_track = ["data science"]
    print("Starting daily data ingestion pipeline...")

    try:
        # --- Load configuration ---
        api_key = _cfg_env_or_config("YOUTUBE_API_KEY")
        mongo_uri = _cfg_env_or_config("MONGO_URI")
        mongo_db_name = _cfg_env_or_config("MONGO_DB_NAME", "trendpulse_youtube_db")

        if not api_key or not mongo_uri:
            raise ValueError("API Key or MongoDB URI is not configured.")

        collector = YouTubeCollector() # Assuming collector is updated to accept the key
        db_handler = MongoHandler() # Assuming handler is updated
        pipeline = [
            {
                "$group": {
                    "_id": "$keyword",           # Group by keyword
                    "count": {"$sum": 1}         # Count occurrences
                }
            },
            {
                "$sort": {"count": -1}           # Optional: Sort by count descending
            }
        ]
        results = db_handler.collection.aggregate(pipeline)
        for doc in results:
            print(f"Keyword: {doc['_id']}, Count: {doc['count']}")
            keywords_to_track.append(doc['_id'])

        # Prepare Reddit collector once if enabled
        reddit_enabled = _cfg_env_or_config("REDDIT_ENABLED", False)
        if isinstance(reddit_enabled, str):
            reddit_enabled = reddit_enabled.lower() in ("1", "true", "yes")
        reddit_client_id = _cfg_env_or_config("REDDIT_CLIENT_ID")
        reddit_col = None
        if reddit_enabled and reddit_client_id:
            try:
                reddit_col = RedditCollector()
            except Exception as e:
                print("Could not initialize RedditCollector (Reddit disabled for this run):", e)
                reddit_col = None

        # --- Ingestion: for each keyword, run both YouTube and Reddit flows ---
        for keyword in keywords_to_track:
            # --- YouTube for keyword ---
            print(f"Fetching data for keyword: '{keyword}' (YouTube)...")
            try:
                data_df = collector.fetch_video_data_by_keyword(keyword, video_limit=20, comment_limit=20)
                if not data_df.empty:
                    print(f"Found {len(data_df)} YouTube comments. Storing in database...")
                    db_handler.insert_data(data_df)
                    print(f"Successfully stored YouTube data for '{keyword}'.")
                else:
                    print(f"No new YouTube data found for '{keyword}'.")
            except Exception as e:
                print(f"YouTube fetch/store failed for '{keyword}': {e}")

            # --- Reddit for the same keyword ---
            if reddit_col is None:
                # skip if reddit not enabled or not initialized
                if reddit_enabled:
                    print(f"Reddit enabled but reddit_col failed to init; skipping Reddit for '{keyword}'.")
                continue

            try:
                print(f"Fetching Reddit posts for keyword: '{keyword}'...")
                df_posts = reddit_col.fetch_posts_by_keyword(keyword, limit=30, sort='relevance', subreddit='all')

                if not df_posts.empty:
                    print(f"Storing {len(df_posts)} posts from Reddit search for '{keyword}' into DB...")

                    # ensure required post/comment columns exist
                    required_cols = ['comment_id']  # keep to satisfy db_handler.schema if needed
                    for col in required_cols:
                        if col not in df_posts.columns:
                            df_posts[col] = None

                    # try high-level insert; fallback to raw insert as before
                    try:
                        db_handler.insert_data(df_posts)
                        print(f"Inserted/Updated {len(df_posts)} Reddit posts for keyword '{keyword}' via db_handler.insert_data().")
                    except Exception as e:
                        print("Primary reddit post insert failed, falling back. Error:", e)
                        try:
                            records = df_posts.to_dict(orient='records')
                            if hasattr(db_handler, "collection") and getattr(db_handler, "collection") is not None:
                                db_handler.collection.insert_many(records)
                                print(f"Inserted {len(records)} Reddit post documents (raw) for keyword '{keyword}'.")
                            else:
                                import pandas as _pd
                                db_handler.insert_data(_pd.DataFrame(records))
                                print(f"Inserted {len(records)} Reddit post documents via fallback for keyword '{keyword}'.")
                        except Exception as e2:
                            print("Fallback raw insert for reddit posts failed:", e2)

                    # For each top post (or first N top posts), fetch comments and insert
                    top_n = min(3, len(df_posts))  # change top_n if you want more/less
                    for i in range(top_n):
                        pid = df_posts.iloc[i]['id']
                        try:
                            comments_df = reddit_col.fetch_comments_for_post(pid, limit=100)
                            if not comments_df.empty:
                                print(f"Storing {len(comments_df)} comments for reddit post {pid} (keyword '{keyword}') into DB...")
                                # ensure comment-required cols
                                comment_required_cols = ['comment_id', 'submission_id', 'parent_id']
                                for col in comment_required_cols:
                                    if col not in comments_df.columns:
                                        comments_df[col] = None
                                try:
                                    db_handler.insert_data(comments_df)
                                    print(f"Inserted/Updated {len(comments_df)} Reddit comments for post {pid}.")
                                except Exception as e:
                                    print("Primary reddit comment insert failed, falling back. Error:", e)
                                    try:
                                        recs = comments_df.to_dict(orient='records')
                                        if hasattr(db_handler, "collection") and getattr(db_handler, "collection") is not None:
                                            db_handler.collection.insert_many(recs)
                                            print(f"Inserted {len(recs)} Reddit comment docs (raw) for post {pid}.")
                                        else:
                                            import pandas as _pd
                                            db_handler.insert_data(_pd.DataFrame(recs))
                                            print(f"Inserted {len(recs)} Reddit comment docs via fallback for post {pid}.")
                                    except Exception as e2:
                                        print("Fallback raw insert for reddit comments failed:", e2)
                        except Exception as e:
                            print(f"Could not fetch comments for reddit post {pid}: {e}")
                else:
                    print(f"No Reddit posts found for keyword '{keyword}'.")
            except Exception as e:
                print(f"Reddit search/insert failed for keyword '{keyword}': {e}")

        # End of keyword loop

    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        # In a real-world scenario, you would add more robust error handling
        # and notifications (e.g., send an email or Slack message).

if __name__ == "__main__":
    run_daily_ingestion()
