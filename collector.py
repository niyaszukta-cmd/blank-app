#!/usr/bin/env python3
"""
================================================================================
NYZTrade - Automated GEX/DEX Data Collector
================================================================================
This script runs independently in the background to collect and store
GEX/DEX data at regular intervals. No browser or login required.

Usage:
    python data_collector.py                    # Run once
    python data_collector.py --continuous       # Run continuously
    python data_collector.py --interval 5       # Custom interval (minutes)

Schedule with cron (Linux/Mac):
    */3 9-16 * * 1-5 /usr/bin/python3 /path/to/data_collector.py

Schedule with Task Scheduler (Windows):
    Create task to run every 3 minutes during market hours
================================================================================
"""

import sqlite3
import json
import os
import sys
import time
import argparse
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try importing the calculator
try:
    from gex_calculator import EnhancedGEXDEXCalculator, calculate_dual_gex_dex_flow, detect_gamma_flip_zones
    CALCULATOR_AVAILABLE = True
except ImportError as e:
    logger.error(f"Cannot import gex_calculator: {e}")
    CALCULATOR_AVAILABLE = False


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DATABASE_FILE = "gex_dex_history.db"

# Symbols to collect
SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]

# Default settings
DEFAULT_STRIKES_RANGE = 12
DEFAULT_EXPIRY_INDEX = 0


# ============================================================================
# DATABASE SETUP
# ============================================================================

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Main snapshots table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            symbol TEXT NOT NULL,
            futures_ltp REAL,
            spot_price REAL,
            fetch_method TEXT,
            total_gex REAL,
            total_dex REAL,
            gex_bias TEXT,
            dex_bias TEXT,
            combined_bias TEXT,
            atm_strike REAL,
            atm_straddle_premium REAL,
            pcr REAL,
            expiry_date TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp, symbol)
        )
    """)
    
    # Strike-wise data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strike_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            strike REAL NOT NULL,
            call_oi INTEGER,
            put_oi INTEGER,
            call_oi_change INTEGER,
            put_oi_change INTEGER,
            call_volume INTEGER,
            put_volume INTEGER,
            call_iv REAL,
            put_iv REAL,
            call_ltp REAL,
            put_ltp REAL,
            net_gex REAL,
            net_dex REAL,
            hedging_pressure REAL,
            FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
        )
    """)
    
    # Flow metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flow_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            gex_near_total REAL,
            gex_near_positive REAL,
            gex_near_negative REAL,
            dex_near_total REAL,
            dex_near_positive REAL,
            dex_near_negative REAL,
            combined_signal REAL,
            max_call_oi_strike REAL,
            max_put_oi_strike REAL,
            FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
        )
    """)
    
    # Gamma flip zones table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gamma_flips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            lower_strike REAL,
            upper_strike REAL,
            flip_type TEXT,
            FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
        )
    """)
    
    # Create indexes for faster queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON snapshots(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_symbol ON snapshots(symbol)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_strike_data_snapshot ON strike_data(snapshot_id)")
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized: {DATABASE_FILE}")


def save_snapshot(symbol, df, futures_ltp, fetch_method, atm_info, flow_metrics, gamma_flips=None):
    """Save a complete snapshot to database"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    try:
        timestamp = datetime.now()
        
        # Calculate totals
        total_gex = float(df['Net_GEX_B'].sum()) if 'Net_GEX_B' in df.columns else 0
        total_dex = float(df['Net_DEX_B'].sum()) if 'Net_DEX_B' in df.columns else 0
        
        # Get biases
        gex_bias = flow_metrics.get('gex_near_bias', 'N/A') if flow_metrics else 'N/A'
        dex_bias = flow_metrics.get('dex_near_bias', 'N/A') if flow_metrics else 'N/A'
        combined_bias = flow_metrics.get('combined_bias', 'N/A') if flow_metrics else 'N/A'
        
        # ATM info
        atm_strike = atm_info.get('atm_strike', 0) if atm_info else 0
        atm_straddle = atm_info.get('atm_straddle_premium', 0) if atm_info else 0
        expiry_date = atm_info.get('expiry_date', '') if atm_info else ''
        
        # PCR
        total_call_oi = df['Call_OI'].sum() if 'Call_OI' in df.columns else 0
        total_put_oi = df['Put_OI'].sum() if 'Put_OI' in df.columns else 0
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        # Spot price (if available)
        spot_price = atm_info.get('spot_price', futures_ltp) if atm_info else futures_ltp
        
        # Insert main snapshot
        cursor.execute("""
            INSERT OR REPLACE INTO snapshots 
            (timestamp, symbol, futures_ltp, spot_price, fetch_method, total_gex, total_dex,
             gex_bias, dex_bias, combined_bias, atm_strike, atm_straddle_premium, pcr, expiry_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, symbol, futures_ltp, spot_price, fetch_method, total_gex, total_dex,
            gex_bias, dex_bias, combined_bias, atm_strike, atm_straddle, pcr, expiry_date
        ))
        
        snapshot_id = cursor.lastrowid
        
        # Insert strike-wise data
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO strike_data 
                (snapshot_id, strike, call_oi, put_oi, call_oi_change, put_oi_change,
                 call_volume, put_volume, call_iv, put_iv, call_ltp, put_ltp,
                 net_gex, net_dex, hedging_pressure)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_id,
                row.get('Strike', 0),
                row.get('Call_OI', 0),
                row.get('Put_OI', 0),
                row.get('Call_OI_Change', 0),
                row.get('Put_OI_Change', 0),
                row.get('Call_Volume', 0),
                row.get('Put_Volume', 0),
                row.get('Call_IV', 0),
                row.get('Put_IV', 0),
                row.get('Call_LTP', 0),
                row.get('Put_LTP', 0),
                row.get('Net_GEX_B', 0),
                row.get('Net_DEX_B', 0),
                row.get('Hedging_Pressure', 0)
            ))
        
        # Insert flow metrics
        if flow_metrics:
            cursor.execute("""
                INSERT INTO flow_metrics 
                (snapshot_id, gex_near_total, gex_near_positive, gex_near_negative,
                 dex_near_total, dex_near_positive, dex_near_negative, combined_signal,
                 max_call_oi_strike, max_put_oi_strike)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_id,
                flow_metrics.get('gex_near_total', 0),
                flow_metrics.get('gex_near_positive', 0),
                flow_metrics.get('gex_near_negative', 0),
                flow_metrics.get('dex_near_total', 0),
                flow_metrics.get('dex_near_positive', 0),
                flow_metrics.get('dex_near_negative', 0),
                flow_metrics.get('combined_signal', 0),
                flow_metrics.get('max_call_oi_strike', 0),
                flow_metrics.get('max_put_oi_strike', 0)
            ))
        
        # Insert gamma flip zones
        if gamma_flips:
            for flip in gamma_flips:
                cursor.execute("""
                    INSERT INTO gamma_flips (snapshot_id, lower_strike, upper_strike, flip_type)
                    VALUES (?, ?, ?, ?)
                """, (
                    snapshot_id,
                    flip.get('lower_strike', 0),
                    flip.get('upper_strike', 0),
                    flip.get('flip_type', 'unknown')
                ))
        
        conn.commit()
        logger.info(f"✅ Saved snapshot for {symbol} at {timestamp.strftime('%H:%M:%S')} (ID: {snapshot_id})")
        return snapshot_id
        
    except Exception as e:
        conn.rollback()
        logger.error(f"❌ Failed to save snapshot for {symbol}: {e}")
        return None
    finally:
        conn.close()


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_data_for_symbol(symbol, strikes_range=DEFAULT_STRIKES_RANGE, expiry_index=DEFAULT_EXPIRY_INDEX):
    """Collect GEX/DEX data for a single symbol"""
    if not CALCULATOR_AVAILABLE:
        logger.error("Calculator not available")
        return False
    
    try:
        calculator = EnhancedGEXDEXCalculator()
        
        # Fetch data
        df, futures_ltp, fetch_method, atm_info = calculator.fetch_and_calculate_gex_dex(
            symbol=symbol,
            strikes_range=strikes_range,
            expiry_index=expiry_index
        )
        
        if df is None or df.empty:
            logger.warning(f"No data received for {symbol}")
            return False
        
        # Calculate flow metrics
        try:
            flow_metrics = calculate_dual_gex_dex_flow(df, futures_ltp)
        except:
            flow_metrics = None
        
        # Detect gamma flips
        try:
            gamma_flips = detect_gamma_flip_zones(df)
        except:
            gamma_flips = None
        
        # Save to database
        snapshot_id = save_snapshot(symbol, df, futures_ltp, fetch_method, atm_info, flow_metrics, gamma_flips)
        
        return snapshot_id is not None
        
    except Exception as e:
        logger.error(f"Error collecting data for {symbol}: {e}")
        return False


def collect_all_symbols():
    """Collect data for all configured symbols"""
    logger.info("=" * 60)
    logger.info(f"Starting data collection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    for symbol in SYMBOLS:
        success = collect_data_for_symbol(symbol)
        results[symbol] = success
        time.sleep(2)  # Small delay between symbols to avoid rate limiting
    
    success_count = sum(results.values())
    logger.info(f"Collection complete: {success_count}/{len(SYMBOLS)} successful")
    logger.info("=" * 60)
    
    return results


def is_market_hours():
    """Check if current time is within market hours (IST)"""
    try:
        import pytz
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
    except:
        # Fallback if pytz not available
        now = datetime.now()
    
    # Market hours: 9:15 AM to 3:30 PM, Monday to Friday
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_open <= now <= market_close


def run_continuous(interval_minutes=3, market_hours_only=True):
    """Run data collection continuously at specified interval"""
    logger.info(f"Starting continuous collection every {interval_minutes} minutes")
    logger.info(f"Market hours only: {market_hours_only}")
    
    while True:
        try:
            if market_hours_only and not is_market_hours():
                logger.info("Outside market hours. Waiting...")
                time.sleep(60)  # Check every minute
                continue
            
            collect_all_symbols()
            
            logger.info(f"Sleeping for {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
            
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in continuous loop: {e}")
            time.sleep(60)  # Wait a minute before retrying


# ============================================================================
# DATABASE QUERY FUNCTIONS (for dashboard to use)
# ============================================================================

def get_available_dates(symbol=None):
    """Get list of dates with available data"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    if symbol:
        cursor.execute("""
            SELECT DISTINCT DATE(timestamp) as date 
            FROM snapshots 
            WHERE symbol = ?
            ORDER BY date DESC
        """, (symbol,))
    else:
        cursor.execute("""
            SELECT DISTINCT DATE(timestamp) as date 
            FROM snapshots 
            ORDER BY date DESC
        """)
    
    dates = [row[0] for row in cursor.fetchall()]
    conn.close()
    return dates


def get_snapshots_for_date(symbol, date):
    """Get all snapshots for a specific symbol and date"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, timestamp, futures_ltp, total_gex, total_dex, gex_bias, dex_bias,
               combined_bias, atm_strike, atm_straddle_premium, pcr
        FROM snapshots 
        WHERE symbol = ? AND DATE(timestamp) = ?
        ORDER BY timestamp ASC
    """, (symbol, date))
    
    columns = ['id', 'timestamp', 'futures_ltp', 'total_gex', 'total_dex', 'gex_bias',
               'dex_bias', 'combined_bias', 'atm_strike', 'atm_straddle_premium', 'pcr']
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(zip(columns, row)) for row in rows]


def get_strike_data(snapshot_id):
    """Get strike-wise data for a specific snapshot"""
    conn = sqlite3.connect(DATABASE_FILE)
    
    import pandas as pd
    df = pd.read_sql_query("""
        SELECT strike as Strike, call_oi as Call_OI, put_oi as Put_OI,
               call_oi_change as Call_OI_Change, put_oi_change as Put_OI_Change,
               call_volume as Call_Volume, put_volume as Put_Volume,
               call_iv as Call_IV, put_iv as Put_IV,
               call_ltp as Call_LTP, put_ltp as Put_LTP,
               net_gex as Net_GEX_B, net_dex as Net_DEX_B,
               hedging_pressure as Hedging_Pressure
        FROM strike_data 
        WHERE snapshot_id = ?
        ORDER BY strike ASC
    """, conn, params=(snapshot_id,))
    
    conn.close()
    return df


def get_flow_metrics(snapshot_id):
    """Get flow metrics for a specific snapshot"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM flow_metrics WHERE snapshot_id = ?
    """, (snapshot_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        columns = ['id', 'snapshot_id', 'gex_near_total', 'gex_near_positive', 'gex_near_negative',
                   'dex_near_total', 'dex_near_positive', 'dex_near_negative', 'combined_signal',
                   'max_call_oi_strike', 'max_put_oi_strike']
        return dict(zip(columns, row))
    return None


def get_intraday_history(symbol, date):
    """Get intraday price and GEX history for charting"""
    conn = sqlite3.connect(DATABASE_FILE)
    
    import pandas as pd
    df = pd.read_sql_query("""
        SELECT timestamp, futures_ltp, total_gex, total_dex, gex_bias, pcr
        FROM snapshots 
        WHERE symbol = ? AND DATE(timestamp) = ?
        ORDER BY timestamp ASC
    """, conn, params=(symbol, date))
    
    conn.close()
    return df


def get_database_stats():
    """Get database statistics"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    stats = {}
    
    # Total snapshots
    cursor.execute("SELECT COUNT(*) FROM snapshots")
    stats['total_snapshots'] = cursor.fetchone()[0]
    
    # Snapshots by symbol
    cursor.execute("SELECT symbol, COUNT(*) FROM snapshots GROUP BY symbol")
    stats['by_symbol'] = dict(cursor.fetchall())
    
    # Date range
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM snapshots")
    row = cursor.fetchone()
    stats['first_snapshot'] = row[0]
    stats['last_snapshot'] = row[1]
    
    # Database size
    stats['db_size_mb'] = os.path.getsize(DATABASE_FILE) / (1024 * 1024) if os.path.exists(DATABASE_FILE) else 0
    
    conn.close()
    return stats


def cleanup_old_data(days_to_keep=30):
    """Remove data older than specified days"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    # Get snapshot IDs to delete
    cursor.execute("SELECT id FROM snapshots WHERE timestamp < ?", (cutoff_date,))
    old_ids = [row[0] for row in cursor.fetchall()]
    
    if old_ids:
        # Delete related data
        cursor.execute(f"DELETE FROM strike_data WHERE snapshot_id IN ({','.join('?' * len(old_ids))})", old_ids)
        cursor.execute(f"DELETE FROM flow_metrics WHERE snapshot_id IN ({','.join('?' * len(old_ids))})", old_ids)
        cursor.execute(f"DELETE FROM gamma_flips WHERE snapshot_id IN ({','.join('?' * len(old_ids))})", old_ids)
        cursor.execute(f"DELETE FROM snapshots WHERE id IN ({','.join('?' * len(old_ids))})", old_ids)
        
        conn.commit()
        logger.info(f"Cleaned up {len(old_ids)} old snapshots")
    
    # Vacuum database to reclaim space
    cursor.execute("VACUUM")
    conn.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='NYZTrade GEX/DEX Data Collector')
    parser.add_argument('--continuous', '-c', action='store_true', help='Run continuously')
    parser.add_argument('--interval', '-i', type=int, default=3, help='Collection interval in minutes')
    parser.add_argument('--symbol', '-s', type=str, help='Collect only specific symbol')
    parser.add_argument('--all-hours', '-a', action='store_true', help='Collect outside market hours too')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--cleanup', type=int, help='Cleanup data older than N days')
    
    args = parser.parse_args()
    
    # Initialize database
    init_database()
    
    # Show stats
    if args.stats:
        stats = get_database_stats()
        print("\n📊 Database Statistics:")
        print(f"   Total Snapshots: {stats['total_snapshots']}")
        print(f"   By Symbol: {stats['by_symbol']}")
        print(f"   First: {stats['first_snapshot']}")
        print(f"   Last: {stats['last_snapshot']}")
        print(f"   Size: {stats['db_size_mb']:.2f} MB")
        return
    
    # Cleanup
    if args.cleanup:
        cleanup_old_data(args.cleanup)
        return
    
    # Collect data
    if args.continuous:
        run_continuous(
            interval_minutes=args.interval,
            market_hours_only=not args.all_hours
        )
    else:
        if args.symbol:
            collect_data_for_symbol(args.symbol)
        else:
            collect_all_symbols()


if __name__ == "__main__":
    main()
