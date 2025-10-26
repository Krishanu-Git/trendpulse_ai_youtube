import os
from src.data_collection.collector import YouTubeCollector
from src.database.db_handler import MongoHandler
from config import YOUTUBE_API_KEY, MONGO_URI, MONGO_DB_NAME

def run_daily_ingestion():
    """
    This function runs the daily data ingestion pipeline.
    It fetches data for a predefined list of keywords and stores it in MongoDB.
    """
    # List of core keywords to track automatically
    keywords_to_track = ["data science"]
    
    print("Starting daily data ingestion pipeline...")
    
    try:
        # Initialize the collector and database handler
        # For automation, we prioritize environment variables over config.py
        api_key = os.getenv('YOUTUBE_API_KEY', YOUTUBE_API_KEY)
        mongo_uri = os.getenv('MONGO_URI', MONGO_URI)
        
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

        for keyword in keywords_to_track:
            print(f"Fetching data for keyword: '{keyword}'...")
            
            # Fetch 20 videos and 20 comments per video
            data_df = collector.fetch_video_data_by_keyword(keyword, video_limit=20, comment_limit=20)
            
            if not data_df.empty:
                print(f"Found {len(data_df)} comments. Storing in database...")
                db_handler.insert_data(data_df)
                print(f"Successfully stored data for '{keyword}'.")
            else:
                print(f"No new data found for '{keyword}'.")

    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        # In a real-world scenario, you would add more robust error handling
        # and notifications (e.g., send an email or Slack message).

if __name__ == "__main__":
    run_daily_ingestion()
