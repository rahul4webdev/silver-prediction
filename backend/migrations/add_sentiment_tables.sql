-- Migration: Add sentiment tables and fix COMEX sync constraint
-- Run this on the server to apply changes

-- 1. Create news_articles table
CREATE TABLE IF NOT EXISTS news_articles (
    id BIGSERIAL PRIMARY KEY,
    url VARCHAR(500) NOT NULL UNIQUE,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    source VARCHAR(100) NOT NULL,
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    fetched_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    asset VARCHAR(20) NOT NULL,
    relevance_score FLOAT NOT NULL DEFAULT 0.0,
    sentiment_score FLOAT NOT NULL DEFAULT 0.0,
    sentiment_label VARCHAR(20) NOT NULL DEFAULT 'neutral',
    keyword_matches TEXT,
    is_processed BOOLEAN DEFAULT TRUE
);

-- Create indexes for news_articles
CREATE INDEX IF NOT EXISTS idx_news_articles_asset_published ON news_articles(asset, published_at);
CREATE INDEX IF NOT EXISTS idx_news_articles_sentiment ON news_articles(asset, sentiment_score);

-- 2. Create sentiment_snapshots table
CREATE TABLE IF NOT EXISTS sentiment_snapshots (
    id BIGSERIAL PRIMARY KEY,
    asset VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    overall_sentiment FLOAT NOT NULL,
    sentiment_label VARCHAR(20) NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 0.0,
    article_count INTEGER NOT NULL DEFAULT 0,
    bullish_count INTEGER NOT NULL DEFAULT 0,
    bearish_count INTEGER NOT NULL DEFAULT 0,
    neutral_count INTEGER NOT NULL DEFAULT 0,
    avg_relevance FLOAT NOT NULL DEFAULT 0.0,
    sentiment_7d_avg FLOAT,
    sentiment_momentum FLOAT,
    google_news_count INTEGER DEFAULT 0,
    newsapi_count INTEGER DEFAULT 0,
    lookback_days INTEGER NOT NULL DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_sentiment_snapshot_asset_time UNIQUE (asset, timestamp)
);

-- Create indexes for sentiment_snapshots
CREATE INDEX IF NOT EXISTS idx_sentiment_snapshot_lookup ON sentiment_snapshots(asset, timestamp);

-- 3. Add COMEX-specific unique constraint for price_data
-- This allows upsert to work correctly for COMEX data (NULL instrument_key)
-- First, check if constraint already exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_price_data_comex'
    ) THEN
        -- Create a partial unique index for COMEX data (instrument_key IS NULL)
        CREATE UNIQUE INDEX uq_price_data_comex
        ON price_data (asset, market, interval, timestamp)
        WHERE instrument_key IS NULL;
    END IF;
END $$;

-- 4. Verify tables created
SELECT
    table_name,
    (SELECT count(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
FROM information_schema.tables t
WHERE table_schema = 'public'
  AND table_name IN ('news_articles', 'sentiment_snapshots');
