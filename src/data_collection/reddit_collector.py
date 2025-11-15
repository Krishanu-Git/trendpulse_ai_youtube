import os
import pandas as pd
from datetime import datetime
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

try:
    import praw
except Exception as e:
    raise ImportError("praw is required for Reddit integration. Add 'praw' to requirements.txt") from e

class RedditCollector:
    """Collects posts and comments from Reddit (lightweight, returns pandas DataFrames)."""

    def __init__(self):
        if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT):
            raise ValueError("Reddit credentials (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT) must be set in config.py")
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )

    def fetch_hot_posts(self, subreddit_name="technology", limit=50):
        posts = []
        subreddit = self.reddit.subreddit(subreddit_name)
        for post in subreddit.hot(limit=limit):
            posts.append({
                "id": post.id,
                "title": post.title,
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": datetime.fromtimestamp(post.created_utc),
                "author": str(post.author) if post.author else "[deleted]",
                "subreddit": post.subreddit.display_name,
                "url": post.url,
                "selftext": post.selftext,
                "is_self": post.is_self,
                "permalink": f"https://reddit.com{post.permalink}",
                "flair_text": post.link_flair_text
            })
        return pd.DataFrame(posts)

    def fetch_posts_by_keyword(self, keyword, limit=50, sort='relevance', subreddit='all'):
        """
        Search Reddit for posts matching a keyword (defaults to r/all).
        Returns a pandas.DataFrame similar to fetch_hot_posts.
        """
        posts = []
        sr = self.reddit.subreddit(subreddit)
        # use praw's subreddit.search; 'sort' may be 'relevance', 'top', 'new', etc.
        for post in sr.search(keyword, limit=limit, sort=sort):
            posts.append({
                "id": post.id,
                "title": post.title,
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": datetime.fromtimestamp(post.created_utc),
                "author": str(post.author) if post.author else "[deleted]",
                "subreddit": post.subreddit.display_name,
                "url": post.url,
                "selftext": post.selftext,
                "is_self": post.is_self,
                "permalink": f"https://reddit.com{post.permalink}",
                "flair_text": post.link_flair_text
            })
        return pd.DataFrame(posts)

    def fetch_comments_for_post(self, post_id, limit=200):
        submission = self.reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)
        comments = []
        for comment in submission.comments.list()[:limit]:
            if hasattr(comment, 'body'):
                comments.append({
                    "comment_id": comment.id,
                    "body": comment.body,
                    "score": comment.score,
                    "created_utc": datetime.fromtimestamp(comment.comment_published_at),
                    "author": str(comment.author) if comment.author else "[deleted]",
                    "parent_id": comment.parent_id,
                    "submission_id": comment.submission.id,
                    "depth": comment.depth
                })
        return pd.DataFrame(comments)
