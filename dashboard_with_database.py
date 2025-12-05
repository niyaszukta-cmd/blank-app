"""
================================================================================
NYZTrade - Live Data Fetcher Module
================================================================================
Robust NSE data fetching with multiple fallback methods:
1. Direct NSE API with proper session handling
2. Groww.in futures price
3. Multiple retry mechanisms
4. Detailed status reporting

Author: NYZTrade
================================================================================
"""

import requests
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
import time
import random

warnings.filterwarnings('ignore')


# ============================================================================
# USER AGENT ROTATION
# ============================================================================

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
]

def get_random_ua():
    return random.choice(USER_AGENTS)


# ============================================================================
# BLACK-SCHOLES CALCULATOR
# ============================================================================

class BlackScholesCalculator:
    """Calculate option Greeks using Black-Scholes model"""
    
    @staticmethod
    def calculate_d1(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            return d1
        except:
            return 0

    @staticmethod
    def calculate_gamma(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            n_prime_d1 = norm.pdf(d1)
            gamma = n_prime_d1 / (S * sigma * np.sqrt(T))
            return gamma
        except:
            return 0

    @staticmethod
    def calculate_call_delta(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            return norm.cdf(d1)
        except:
            return 0

    @staticmethod
    def calculate_put_delta(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        try:
            d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
            return norm.cdf(d1) - 1
        except:
            return 0


# ============================================================================
# NSE DATA FETCHER - ROBUST VERSION
# ============================================================================

class RobustNSEFetcher:
    """
    Robust NSE Option Chain Fetcher with:
    - Proper session/cookie handling
    - User agent rotation
    - Multiple retry attempts
    - Detailed status reporting
    """
    
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.option_chain_url = "https://www.nseindia.com/api/option-chain-indices"
        self.session = None
        self.cookies_valid = False
        self.last_init_time = None
        self.status_log = []
        
    def log_status(self, message, level="INFO"):
        """Log status for debugging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_log.append(f"[{timestamp}] [{level}] {message}")
        if len(self.status_log) > 50:
            self.status_log = self.status_log[-50:]
    
    def get_status_log(self):
        """Get recent status messages"""
        return self.status_log[-10:]
    
    def create_session(self):
        """Create new session with fresh cookies"""
        self.session = requests.Session()
        
        headers = {
            'User-Agent': get_random_ua(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
        }
        
        self.session.headers.update(headers)
        return self.session
    
    def initialize_session(self, max_retries=3):
        """Initialize session with NSE website to get cookies"""
        
        for attempt in range(max_retries):
            try:
                self.log_status(f"Initializing NSE session (attempt {attempt + 1}/{max_retries})")
                
                # Create fresh session
                self.create_session()
                
                # First, visit the main page
                response = self.session.get(
                    self.base_url,
                    timeout=15,
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    self.log_status(f"Main page OK, cookies: {len(self.session.cookies)}")
                    
                    # Small delay to mimic human behavior
                    time.sleep(0.5 + random.random())
                    
                    # Visit option chain page to get additional cookies
                    oc_page = self.session.get(
                        f"{self.base_url}/option-chain",
                        timeout=15
                    )
                    
                    if oc_page.status_code == 200:
                        self.log_status("Option chain page OK")
                        
                        # Update headers for API calls
                        self.session.headers.update({
                            'Accept': 'application/json, text/plain, */*',
                            'Referer': 'https://www.nseindia.com/option-chain',
                            'X-Requested-With': 'XMLHttpRequest',
                        })
                        
                        self.cookies_valid = True
                        self.last_init_time = datetime.now()
                        self.log_status("Session initialized successfully", "SUCCESS")
                        return True, "Session initialized"
                
                self.log_status(f"Attempt {attempt + 1} failed: Status {response.status_code}", "WARNING")
                time.sleep(1 + random.random())
                
            except requests.exceptions.Timeout:
                self.log_status(f"Attempt {attempt + 1}: Timeout", "WARNING")
            except requests.exceptions.ConnectionError as e:
                self.log_status(f"Attempt {attempt + 1}: Connection error", "WARNING")
            except Exception as e:
                self.log_status(f"Attempt {attempt + 1}: {str(e)[:50]}", "ERROR")
        
        self.cookies_valid = False
        return False, "Failed to initialize session after all retries"
    
    def fetch_option_chain(self, symbol="NIFTY", max_retries=3):
        """Fetch option chain data with retries"""
        
        # Check if session needs refresh (every 3 minutes)
        if self.last_init_time:
            elapsed = (datetime.now() - self.last_init_time).total_seconds()
            if elapsed > 180:
                self.cookies_valid = False
        
        # Initialize if needed
        if not self.cookies_valid or self.session is None:
            success, msg = self.initialize_session()
            if not success:
                return None, msg
        
        url = f"{self.option_chain_url}?symbol={symbol}"
        
        for attempt in range(max_retries):
            try:
                self.log_status(f"Fetching {symbol} option chain (attempt {attempt + 1})")
                
                response = self.session.get(url, timeout=15)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        if 'records' in data and 'data' in data['records']:
                            records = data['records']
                            strikes_count = len(records.get('data', []))
                            spot = records.get('underlyingValue', 0)
                            
                            self.log_status(f"SUCCESS: {symbol} - {strikes_count} strikes, Spot: {spot}", "SUCCESS")
                            return data, None
                        else:
                            self.log_status("Invalid response format", "WARNING")
                            
                    except json.JSONDecodeError:
                        self.log_status("JSON decode error", "WARNING")
                
                elif response.status_code == 401:
                    self.log_status("401 Unauthorized - Refreshing session", "WARNING")
                    self.cookies_valid = False
                    success, msg = self.initialize_session()
                    if not success:
                        return None, msg
                
                elif response.status_code == 403:
                    self.log_status("403 Forbidden - IP might be blocked", "ERROR")
                    return None, "NSE blocked request (403). Try running locally."
                
                else:
                    self.log_status(f"HTTP {response.status_code}", "WARNING")
                
                time.sleep(1 + random.random())
                
            except requests.exceptions.Timeout:
                self.log_status(f"Timeout on attempt {attempt + 1}", "WARNING")
            except Exception as e:
                self.log_status(f"Error: {str(e)[:50]}", "ERROR")
        
        return None, f"Failed to fetch {symbol} after {max_retries} attempts"


# ============================================================================
# GROWW FUTURES FETCHER
# ============================================================================

class GrowwFuturesFetcher:
    """Fetch futures price from Groww.in"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': get_random_ua(),
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://groww.in',
            'Referer': 'https://groww.in/derivatives',
        }
    
    def get_futures_price(self, symbol, spot_price=None):
        """
        Fetch futures price from Groww.in
        Returns: (price, method) or (None, error)
        """
        
        # Method 1: Groww derivatives API
        price = self._try_groww_api(symbol)
        if price:
            return price, "Groww API"
        
        # Method 2: Groww search API
        price = self._try_groww_search(symbol)
        if price:
            return price, "Groww Search"
        
        # Method 3: Calculate from spot (if provided)
        if spot_price:
            # Add typical futures premium (~0.05-0.1%)
            futures = spot_price * 1.0005
            return round(futures, 2), "Spot+Premium"
        
        return None, "Groww fetch failed"
    
    def _try_groww_api(self, symbol):
        """Try Groww derivatives API"""
        try:
            symbol_map = {
                'NIFTY': 'NIFTY',
                'BANKNIFTY': 'BANKNIFTY',
                'FINNIFTY': 'FINNIFTY',
                'MIDCPNIFTY': 'MIDCPNIFTY'
            }
            
            groww_symbol = symbol_map.get(symbol, symbol)
            
            # Try futures contracts endpoint
            url = f"https://groww.in/v1/api/stocks_fo_data/v1/derivatives/futures/contracts/{groww_symbol}"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list) and len(data) > 0:
                    for contract in data:
                        if 'ltp' in contract:
                            return float(contract['ltp'])
                        if 'lastPrice' in contract:
                            return float(contract['lastPrice'])
                
                if isinstance(data, dict):
                    if 'ltp' in data:
                        return float(data['ltp'])
            
            return None
        except:
            return None
    
    def _try_groww_search(self, symbol):
        """Try Groww search endpoint"""
        try:
            url = f"https://groww.in/v1/api/search/v1/entity?page=0&query={symbol}%20FUT&size=10"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                content = data.get('content', [])
                for item in content:
                    if 'FUT' in item.get('title', '').upper():
                        ltp = item.get('ltp') or item.get('lastPrice')
                        if ltp:
                            return float(ltp)
            
            return None
        except:
            return None


# ============================================================================
# MAIN CALCULATOR
# ============================================================================

class LiveGEXDEXCalculator:
    """
    Live GEX + DEX Calculator with:
    - Robust NSE fetching
    - Groww futures price
    - Proper error handling
    - Status reporting
    """
    
    def __init__(self):
        self.nse_fetcher = RobustNSEFetcher()
        self.groww_fetcher = GrowwFuturesFetcher()
        self.bs_calc = BlackScholesCalculator()
        self.risk_free_rate = 0.07
        self.last_error = None
        self.data_source = "Unknown"
    
    def get_contract_specs(self, symbol):
        """Get contract specifications"""
        specs = {
            'NIFTY': {'lot_size': 25, 'strike_interval': 50},
            'BANKNIFTY': {'lot_size': 15, 'strike_interval': 100},
            'FINNIFTY': {'lot_size': 40, 'strike_interval': 50},
            'MIDCPNIFTY': {'lot_size': 75, 'strike_interval': 25}
        }
        return specs.get(symbol, specs['NIFTY'])
    
    def calculate_time_to_expiry(self, expiry_str):
        """Calculate time to expiry in years"""
        try:
            expiry = datetime.strptime(expiry_str, "%d-%b-%Y")
            now = datetime.now()
            
            # Add time to end of day
            expiry = expiry.replace(hour=15, minute=30)
            
            diff = expiry - now
            days = diff.total_seconds() / (24 * 3600)
            T = max(days / 365, 0.5/365)  # Minimum half day
            return T, max(int(days), 1)
        except:
            return 7/365, 7
    
    def get_status_log(self):
        """Get status log from NSE fetcher"""
        return self.nse_fetcher.get_status_log()
    
    def fetch_live_data(self, symbol="NIFTY", strikes_range=12, expiry_index=0):
        """
        Fetch live GEX/DEX data
        
        Returns:
            tuple: (df, futures_ltp, fetch_method, atm_info, error_message)
        """
        
        self.last_error = None
        
        # Step 1: Fetch NSE option chain
        data, error = self.nse_fetcher.fetch_option_chain(symbol)
        
        if error or not data:
            self.last_error = error or "No data received"
            self.data_source = "Failed"
            return None, None, None, None, self.last_error
        
        try:
            records = data['records']
            spot_price = records.get('underlyingValue', 0)
            timestamp = records.get('timestamp', datetime.now().strftime('%d-%b-%Y %H:%M:%S'))
            expiry_dates = records.get('expiryDates', [])
            
            if not expiry_dates or spot_price == 0:
                self.last_error = "Invalid data: No expiries or spot price"
                return None, None, None, None, self.last_error
            
            # Select expiry
            selected_expiry = expiry_dates[min(expiry_index, len(expiry_dates) - 1)]
            T, days_to_expiry = self.calculate_time_to_expiry(selected_expiry)
            
            # Step 2: Get futures price
            futures_ltp, fetch_method = self.groww_fetcher.get_futures_price(symbol, spot_price)
            
            if not futures_ltp:
                # Fallback: Use spot + premium
                futures_ltp = spot_price * 1.0003
                fetch_method = "Spot+Premium"
            
            self.data_source = f"NSE Live + {fetch_method}"
            
            # Get specs
            specs = self.get_contract_specs(symbol)
            lot_size = specs['lot_size']
            strike_interval = specs['strike_interval']
            
            # Process strikes
            all_strikes = []
            processed = set()
            atm_strike = None
            min_diff = float('inf')
            atm_call_premium = 0
            atm_put_premium = 0
            
            for item in records.get('data', []):
                if item.get('expiryDate') != selected_expiry:
                    continue
                
                strike = item.get('strikePrice', 0)
                if strike == 0 or strike in processed:
                    continue
                
                processed.add(strike)
                
                # Filter by range
                distance = abs(strike - futures_ltp) / strike_interval
                if distance > strikes_range:
                    continue
                
                ce = item.get('CE', {})
                pe = item.get('PE', {})
                
                # Extract data
                call_oi = ce.get('openInterest', 0) or 0
                put_oi = pe.get('openInterest', 0) or 0
                call_oi_change = ce.get('changeinOpenInterest', 0) or 0
                put_oi_change = pe.get('changeinOpenInterest', 0) or 0
                call_volume = ce.get('totalTradedVolume', 0) or 0
                put_volume = pe.get('totalTradedVolume', 0) or 0
                call_iv = ce.get('impliedVolatility', 0) or 15
                put_iv = pe.get('impliedVolatility', 0) or 15
                call_ltp = ce.get('lastPrice', 0) or 0
                put_ltp = pe.get('lastPrice', 0) or 0
                
                # Track ATM
                diff = abs(strike - futures_ltp)
                if diff < min_diff:
                    min_diff = diff
                    atm_strike = strike
                    atm_call_premium = call_ltp
                    atm_put_premium = put_ltp
                
                # Calculate Greeks
                call_iv_dec = max(call_iv / 100, 0.05)
                put_iv_dec = max(put_iv / 100, 0.05)
                
                call_gamma = self.bs_calc.calculate_gamma(futures_ltp, strike, T, self.risk_free_rate, call_iv_dec)
                put_gamma = self.bs_calc.calculate_gamma(futures_ltp, strike, T, self.risk_free_rate, put_iv_dec)
                call_delta = self.bs_calc.calculate_call_delta(futures_ltp, strike, T, self.risk_free_rate, call_iv_dec)
                put_delta = self.bs_calc.calculate_put_delta(futures_ltp, strike, T, self.risk_free_rate, put_iv_dec)
                
                # GEX calculation (in Billions)
                gex_mult = futures_ltp * futures_ltp * lot_size / 1_000_000_000
                call_gex = call_oi * call_gamma * gex_mult
                put_gex = -put_oi * put_gamma * gex_mult
                
                # DEX calculation (in Billions)
                dex_mult = futures_ltp * lot_size / 1_000_000_000
                call_dex = call_oi * call_delta * dex_mult
                put_dex = put_oi * put_delta * dex_mult
                
                all_strikes.append({
                    'Strike': strike,
                    'Call_OI': call_oi,
                    'Put_OI': put_oi,
                    'Call_OI_Change': call_oi_change,
                    'Put_OI_Change': put_oi_change,
                    'Call_Volume': call_volume,
                    'Put_Volume': put_volume,
                    'Call_IV': call_iv,
                    'Put_IV': put_iv,
                    'Call_LTP': call_ltp,
                    'Put_LTP': put_ltp,
                    'Call_Gamma': call_gamma,
                    'Put_Gamma': put_gamma,
                    'Call_Delta': call_delta,
                    'Put_Delta': put_delta,
                    'Call_GEX': call_gex,
                    'Put_GEX': put_gex,
                    'Net_GEX': call_gex + put_gex,
                    'Call_DEX': call_dex,
                    'Put_DEX': put_dex,
                    'Net_DEX': call_dex + put_dex,
                })
            
            if not all_strikes:
                self.last_error = "No strikes found for selected expiry"
                return None, None, None, None, self.last_error
            
            # Create DataFrame
            df = pd.DataFrame(all_strikes).sort_values('Strike').reset_index(drop=True)
            
            # Add _B suffix columns for compatibility
            for col in ['Call_GEX', 'Put_GEX', 'Net_GEX', 'Call_DEX', 'Put_DEX', 'Net_DEX']:
                df[f'{col}_B'] = df[col]
            
            df['Total_Volume'] = df['Call_Volume'] + df['Put_Volume']
            df['Total_OI'] = df['Call_OI'] + df['Put_OI']
            
            # Hedging Pressure
            max_gex = df['Net_GEX_B'].abs().max()
            df['Hedging_Pressure'] = (df['Net_GEX_B'] / max_gex * 100) if max_gex > 0 else 0
            
            # ATM info
            atm_info = {
                'atm_strike': atm_strike or df.iloc[len(df)//2]['Strike'],
                'atm_call_premium': atm_call_premium,
                'atm_put_premium': atm_put_premium,
                'atm_straddle_premium': atm_call_premium + atm_put_premium,
                'spot_price': spot_price,
                'expiry_date': selected_expiry,
                'days_to_expiry': days_to_expiry,
                'timestamp': timestamp,
                'expiry_index': expiry_index,
                'all_expiries': expiry_dates[:5]  # First 5 expiries
            }
            
            return df, futures_ltp, self.data_source, atm_info, None
            
        except Exception as e:
            self.last_error = f"Processing error: {str(e)}"
            return None, None, None, None, self.last_error


# ============================================================================
# FLOW METRICS - VOLATILITY TERMINOLOGY
# ============================================================================

def calculate_flow_metrics(df, futures_ltp):
    """
    Calculate GEX/DEX flow metrics with volatility terminology
    
    Positive GEX = Volatility Dampening
    Negative GEX = Volatility Amplifying
    """
    
    df_unique = df.drop_duplicates(subset=['Strike']).sort_values('Strike').reset_index(drop=True)
    
    # Near-term GEX (5 positive + 5 negative closest to spot)
    pos_gex = df_unique[df_unique['Net_GEX_B'] > 0].copy()
    if len(pos_gex) > 0:
        pos_gex['Dist'] = abs(pos_gex['Strike'] - futures_ltp)
        pos_gex = pos_gex.nsmallest(5, 'Dist')
    
    neg_gex = df_unique[df_unique['Net_GEX_B'] < 0].copy()
    if len(neg_gex) > 0:
        neg_gex['Dist'] = abs(neg_gex['Strike'] - futures_ltp)
        neg_gex = neg_gex.nsmallest(5, 'Dist')
    
    gex_near_pos = float(pos_gex['Net_GEX_B'].sum()) if len(pos_gex) > 0 else 0
    gex_near_neg = float(neg_gex['Net_GEX_B'].sum()) if len(neg_gex) > 0 else 0
    gex_near_total = gex_near_pos + gex_near_neg
    
    # Total GEX
    gex_total = float(df_unique['Net_GEX_B'].sum())
    
    # DEX flow (strikes above/below spot)
    above = df_unique[df_unique['Strike'] > futures_ltp].head(5)
    below = df_unique[df_unique['Strike'] < futures_ltp].tail(5)
    
    dex_near_pos = float(above['Net_DEX_B'].sum()) if len(above) > 0 else 0
    dex_near_neg = float(below['Net_DEX_B'].sum()) if len(below) > 0 else 0
    dex_near_total = dex_near_pos + dex_near_neg
    
    # Total DEX
    dex_total = float(df_unique['Net_DEX_B'].sum())
    
    # Key levels
    max_call_oi_strike = float(df_unique.loc[df_unique['Call_OI'].idxmax(), 'Strike'])
    max_put_oi_strike = float(df_unique.loc[df_unique['Put_OI'].idxmax(), 'Strike'])
    
    # PCR
    total_call_oi = df_unique['Call_OI'].sum()
    total_put_oi = df_unique['Put_OI'].sum()
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1
    
    # Bias determination with VOLATILITY terminology
    def get_gex_bias(val):
        if val > 100:
            return "🟢 STRONG DAMPENING", "#00d4aa"
        elif val > 0:
            return "🟢 DAMPENING", "#55efc4"
        elif val > -100:
            return "🔴 AMPLIFYING", "#ff6b6b"
        else:
            return "🔴 STRONG AMPLIFYING", "#e74c3c"
    
    def get_dex_bias(val):
        if val > 50:
            return "🟢 STRONG BULLISH", "#00d4aa"
        elif val > 0:
            return "🟢 BULLISH", "#55efc4"
        elif val > -50:
            return "🔴 BEARISH", "#ff6b6b"
        else:
            return "🔴 STRONG BEARISH", "#e74c3c"
    
    gex_bias, gex_color = get_gex_bias(gex_near_total)
    dex_bias, dex_color = get_dex_bias(dex_near_total)
    
    # Combined signal
    combined = (gex_near_total + dex_near_total) / 2
    if gex_near_total > 50 and dex_near_total > 20:
        combined_bias = "🟢 DAMPENING + BULLISH"
    elif gex_near_total > 50 and dex_near_total < -20:
        combined_bias = "🟡 DAMPENING + BEARISH"
    elif gex_near_total < -50 and dex_near_total > 20:
        combined_bias = "⚡ AMPLIFYING + BULLISH"
    elif gex_near_total < -50 and dex_near_total < -20:
        combined_bias = "🔴 AMPLIFYING + BEARISH"
    else:
        combined_bias = "⚪ NEUTRAL"
    
    return {
        'gex_near_total': gex_near_total,
        'gex_near_positive': gex_near_pos,
        'gex_near_negative': gex_near_neg,
        'gex_total': gex_total,
        'gex_bias': gex_bias,
        'gex_color': gex_color,
        'dex_near_total': dex_near_total,
        'dex_total': dex_total,
        'dex_bias': dex_bias,
        'dex_color': dex_color,
        'combined_signal': combined,
        'combined_bias': combined_bias,
        'max_call_oi_strike': max_call_oi_strike,
        'max_put_oi_strike': max_put_oi_strike,
        'pcr': pcr,
        'total_call_oi': total_call_oi,
        'total_put_oi': total_put_oi
    }


def detect_gamma_flips(df):
    """Detect gamma flip zones"""
    flips = []
    df_sorted = df.sort_values('Strike').reset_index(drop=True)
    
    for i in range(len(df_sorted) - 1):
        curr = df_sorted.loc[i, 'Net_GEX_B']
        next_val = df_sorted.loc[i + 1, 'Net_GEX_B']
        
        if (curr > 0 and next_val < 0) or (curr < 0 and next_val > 0):
            flips.append({
                'lower': df_sorted.loc[i, 'Strike'],
                'upper': df_sorted.loc[i + 1, 'Strike'],
                'type': "DAMPENING → AMPLIFYING" if curr > 0 else "AMPLIFYING → DAMPENING"
            })
    
    return flips


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NYZTrade - Live Data Test")
    print("=" * 60)
    
    calc = LiveGEXDEXCalculator()
    
    df, futures, method, atm, error = calc.fetch_live_data("NIFTY", 10, 0)
    
    print("\nStatus Log:")
    for log in calc.get_status_log():
        print(f"  {log}")
    
    if error:
        print(f"\n❌ Error: {error}")
    else:
        print(f"\n✅ Success!")
        print(f"   Futures: ₹{futures:,.2f}")
        print(f"   Method: {method}")
        print(f"   Strikes: {len(df)}")
        print(f"   ATM: {atm['atm_strike']}")
        print(f"   Straddle: ₹{atm['atm_straddle_premium']:.2f}")
        
        flow = calculate_flow_metrics(df, futures)
        print(f"   GEX Bias: {flow['gex_bias']}")
        print(f"   DEX Bias: {flow['dex_bias']}")
