# Services module

# Optional imports for sentiment analysis
try:
    from app.services.news_sentiment import news_sentiment_service, NewsSentimentService
except ImportError:
    news_sentiment_service = None
    NewsSentimentService = None
