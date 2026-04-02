import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np

class NewsTemporalAligner:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.text_dir = self.data_dir / "text"
        self.timeseries_dir = self.data_dir / "time_series"
        
    def load_news(self, ticker: str) -> pd.DataFrame:
        """Load news articles for a given ticker symbol."""
        news_file = self.text_dir / f"{ticker.upper()}.jsonl"
        
        if not news_file.exists():
            return pd.DataFrame()
        
        news_list = []
        with open(news_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line)
                    article['Date'] = pd.to_datetime(article['Date'])
                    news_list.append(article)
                except json.JSONDecodeError:
                    continue
        
        return pd.DataFrame(news_list)
    
    def load_timeseries(self, ticker: str) -> pd.DataFrame:
        """Load time series data for a given ticker symbol."""
        ts_file = self.timeseries_dir / f"{ticker.lower()}.csv"
        
        if not ts_file.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(ts_file)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df['Date'] = df['Date'].dt.tz_localize(None)
        df['DateOnly'] = df['Date'].dt.date
        
        return df.sort_values('Date').reset_index(drop=True)
    
    def align_news_to_timeseries(self, ticker: str, lookback_days: int = 5) -> Tuple[pd.DataFrame, Dict]:
        """
        Align news articles to time series data.
        For each article date, find corresponding time series data within lookback window.
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days to look back for time series data from article date
        
        Returns:
            Aligned DataFrame and alignment statistics
        """
        news_df = self.load_news(ticker)
        ts_df = self.load_timeseries(ticker)
        
        if news_df.empty or ts_df.empty:
            return pd.DataFrame(), {"status": "No data found", "ticker": ticker}
        
        # Create alignment results
        aligned_records = []
        stats = {
            "ticker": ticker,
            "total_news": len(news_df),
            "total_timeseries": len(ts_df),
            "news_date_range": (news_df['Date'].min(), news_df['Date'].max()),
            "ts_date_range": (ts_df['Date'].min(), ts_df['Date'].max()),
            "aligned_count": 0,
            "unaligned_news": 0,
            "lookback_days": lookback_days
        }
        
        for _, news_row in news_df.iterrows():
            news_date = news_row['Date']
            
            # Find time series data within lookback window
            lookback_start = news_date - timedelta(days=lookback_days)
            ts_window = ts_df[
                (ts_df['Date'] >= lookback_start) & 
                (ts_df['Date'] <= news_date)
            ]
            
            if not ts_window.empty:
                # Get the nearest trading day before or on the news date
                nearest_ts = ts_window.iloc[-1]  # Most recent data in window
                
                aligned_records.append({
                    'news_date': news_date,
                    'ts_date': nearest_ts['Date'],
                    'days_diff': (news_date - nearest_ts['Date']).days,
                    'stock_symbol': news_row['Stock_symbol'],
                    'article': news_row['Article'],
                    'url': news_row.get('Url', ''),
                    'article_title': news_row.get('Article_title', ''),
                    'open': nearest_ts['Open'],
                    'high': nearest_ts['High'],
                    'low': nearest_ts['Low'],
                    'close': nearest_ts['Close'],
                    'volume': nearest_ts['Volume'],
                })
                stats["aligned_count"] += 1
            else:
                stats["unaligned_news"] += 1
        
        aligned_df = pd.DataFrame(aligned_records)
        return aligned_df, stats
    
    def align_all_tickers(self, lookback_days: int = 5, limit: int = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Align news and time series for all available tickers.
        
        Args:
            lookback_days: Number of days to look back for time series data
            limit: Maximum number of unique tickers to process (for testing)
        
        Returns:
            Combined aligned DataFrame and overall statistics
        """
        # Get all news files
        news_files = list(self.text_dir.glob("*.jsonl"))
        tickers = sorted(set(f.stem.lower() for f in news_files))
        
        if limit:
            tickers = tickers[:limit]
        
        all_aligned = []
        all_stats = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"Processing {ticker} ({i}/{len(tickers)})")
            aligned_df, stats = self.align_news_to_timeseries(ticker, lookback_days)
            
            if not aligned_df.empty:
                all_aligned.append(aligned_df)
            all_stats.append(stats)
        
        combined_df = pd.concat(all_aligned, ignore_index=True) if all_aligned else pd.DataFrame()
        
        # Summary statistics
        summary = {
            "total_tickers": len(tickers),
            "total_news_articles": sum(s.get("total_news", 0) for s in all_stats),
            "total_timeseries_points": sum(s.get("total_timeseries", 0) for s in all_stats),
            "total_aligned": sum(s.get("aligned_count", 0) for s in all_stats),
            "total_unaligned": sum(s.get("unaligned_news", 0) for s in all_stats),
            "lookback_days": lookback_days,
            "per_ticker_stats": all_stats
        }
        
        return combined_df, summary
    
    def save_aligned_data(self, output_path: str = "data/aligned_news_timeseries.csv", limit: int = 50):
        """Save aligned news and time series data."""
        print("Loading and aligning all data...")
        aligned_df, stats = self.align_all_tickers(limit=limit)
        
        if not aligned_df.empty:
            aligned_df.to_csv(output_path, index=False)
            print(f"✓ Saved {len(aligned_df)} aligned records to {output_path}")
        
        print("\n=== Alignment Summary ===")
        print(f"Total tickers processed: {stats['total_tickers']}")
        print(f"Total news articles: {stats['total_news_articles']}")
        print(f"Total time series points: {stats['total_timeseries_points']}")
        print(f"Successfully aligned: {stats['total_aligned']}")
        print(f"Could not align: {stats['total_unaligned']}")
        print(f"Alignment rate: {100 * stats['total_aligned'] / max(stats['total_news_articles'], 1):.1f}%")
        
        return aligned_df, stats


if __name__ == "__main__":
    aligner = NewsTemporalAligner()
    # Test with first 50 tickers, set limit=None to process all
    aligned_df, summary = aligner.save_aligned_data()
    
    # Show sample
    if not aligned_df.empty:
        print("\n=== Sample Aligned Data ===")
        print(aligned_df[['news_date', 'ts_date', 'stock_symbol', 'close', 'volume']].head(10))
