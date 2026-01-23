# Services module

# Optional imports for sentiment analysis
try:
    from app.services.news_sentiment import news_sentiment_service, NewsSentimentService
except ImportError:
    news_sentiment_service = None
    NewsSentimentService = None

# Macro data service
try:
    from app.services.macro_data import macro_data_service, MacroDataService
except ImportError:
    macro_data_service = None
    MacroDataService = None

# Notification service
try:
    from app.services.notifications import telegram_service, TelegramNotificationService
except ImportError:
    telegram_service = None
    TelegramNotificationService = None

# Confluence and correlation services
try:
    from app.services.confluence import (
        confluence_detector,
        correlation_analyzer,
        ConfluenceDetector,
        CorrelationAnalyzer,
    )
except ImportError:
    confluence_detector = None
    correlation_analyzer = None
    ConfluenceDetector = None
    CorrelationAnalyzer = None
