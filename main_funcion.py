import ccxt
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta, timezone
from colorama import Fore, Style, init
import time, os



#12/31 160:52%  100:47%  80:39%  50:42%  45:42%  44:42%  43:42%  


start_date ="2026-01-06 17:00"
# 初始化顏色系統
init(autoreset=True)

class BTCBacktesterUTC8:
    def __init__(self, api_key="", secret_key=""):
        self.SYMBOL = 'BTC/USDT'
        self.TIMEFRAME = '5m'
        self.INIT_CAPITAL = 300.0   
        self.TRADE_QTY_USD = 1500.0 
        self.COMMISSION_PCT = 0.0004 # 單邊 0.04%
        self.CLIMAX_MULT = 4.5
        self.LOOKBACK15, self.LOOKBACK5 = 50, 10# 回溯週期
        self.ADX_THRESHOLD = 25.15
        self.BBW_THRESHOLD = 0.0039
        self.TZ_OFFSET = timedelta(hours=8)
       
        self.SL_PCT = 1.5
        self.TP_PCT = 4.0
        self.MIN_MOVE_PTS = 900.0
        self.SOLID_RATIO = 0.70
        self.er_length = 18
        self.efficiencyratio_threshold = 0.10
        self.r2_length = 10
        self.r2_threshold = 0.20
        self.MAX_LOSS_PTS = 200.00
        self.last_exit_bar_time = None  

       # ------------------------------------------------------------------
       # 【新增：行情數據源】專門抓取正式盤的「真實價格」，不需要 Key
        self.data_exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        #-------------------------------------------------------------------



        # --- 交易所初始化 (加入 Key 判斷與 Demo Trading 支援) ---
        is_real_key = api_key and all(word not in api_key for word in ["KEY", "你的", "YOUR"])
        config = {
            'apiKey': api_key.strip() if is_real_key else "",
            'secret': secret_key.strip() if is_real_key else "",
            'enableRateLimit': True,
            'adjustForTimeDifference': True,
            'recvWindow': 10000,
            'options': {'defaultType': 'future'}
        }
        self.exchange = ccxt.binance(config)
        
        if is_real_key:
        
            self.exchange.enable_demo_trading(True)
            
            try:
                self.exchange.set_leverage(5, self.SYMBOL)   #============================================杠桿調整=========================================
            except: pass
            print(f"{Fore.CYAN} 已啟用 實時下單模式")

    def fetch_data(self,limit =100):                                                      #start_str                                                                                                                                                                                                      #第二個變數：抓取最近的k棒數 #start_str
        local_time = datetime.strptime(start_date , "%Y-%m-%d %H:%M")
        utc_time = local_time - self.TZ_OFFSET
        since = int(utc_time.replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        all_bars = []
        print(f"抓取數據 (時區: UTC+8, 起點: {start_date})...")
        while since < int(datetime.now(timezone.utc).timestamp() * 1000):
            try:
                #----------------------------------------------------------------------------
                bars = self.data_exchange.fetch_ohlcv(self.SYMBOL, timeframe=self.TIMEFRAME, since=since, limit=1000)# bars = self.exchange.fetch_ohlcv(self.SYMBOL, timeframe=self.TIMEFRAME, since=since, limit=1000)
                #-----------------------------------------------------------------------------
                if not bars: break
                since = bars[-1][0] + 1
                all_bars += bars
                if len(bars) < 1000: break
            except: break
        df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol'])
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms') + self.TZ_OFFSET
        return df

    def run_simulation(self, df):
                                                                                                                                                                                                                                                            # --- 指標計算 (100% 同步 Pine Script) ---
        # --- ⚙️ 指標計算 (100% 同步 Pine Script) ---
        df['body'] = (df['close'] - df['open']).abs()
        df['avg_body'] = df['body'].rolling(20).mean()
        df['upper15'] = df['high'].shift(1).rolling(self.LOOKBACK15).max()
        df['lower15'] = df['low'].shift(1).rolling(self.LOOKBACK15).min()
        df['m5_hh'] = df['high'].shift(1).rolling(self.LOOKBACK5).max()
        df['m5_ll'] = df['low'].shift(1).rolling(self.LOOKBACK5).min()
        
        # 1. ADX 與 BBW
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx_v'] = adx_df['ADX_14']
        df['bbw_v'] = (df['high'].rolling(14).max() - df['low'].rolling(14).min()) / df['close'].rolling(14).mean()
        df['wide_vol_ok'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) > 150

        # 2. Efficiency Ratio (效率比)
        net_chg = (df['close'] - df['close'].shift(self.er_length)).abs()
        total_chg = (df['close'] - df['close'].shift(1)).abs().rolling(self.er_length).sum()
        df['er_v'] = net_chg / total_chg
        df['isEfficient'] = df['er_v'] > self.efficiencyratio_threshold

        # 3. R-Squared (線性回歸 R 平方)
        # 建立時間序列索引用於計算相關係數
        df['bar_idx'] = range(len(df))
        df['r_squared'] = df['close'].rolling(self.r2_length).corr(df['bar_idx']) ** 2
        df['is_linear_trend'] = df['r_squared'] > self.r2_threshold

        def is_solid(row):
            f = row['high'] - row['low']
            return (row['body'] / f >= self.SOLID_RATIO) if f > 0 else False
        
        df['is_solid'] = df.apply(is_solid, axis=1)
        df['is_rev_r'] = df['is_solid'] & (df['close'] < df['open']) & (df['body'] > df['avg_body'] * self.CLIMAX_MULT)
        df['is_rev_g'] = df['is_solid'] & (df['close'] > df['open']) & (df['body'] > df['avg_body'] * self.CLIMAX_MULT)
        df['loss_l'] = (df['close'] < df['open']) & df['is_solid'] & (df['close'].shift(1) < df['open'].shift(1)) & df['is_solid'].shift(1)
        df['loss_s'] = (df['close'] > df['open']) & df['is_solid'] & (df['close'].shift(1) > df['open'].shift(1)) & df['is_solid'].shift(1)

        pos, entry_p, entry_t, entry_idx = 0, 0.0, None, -1
        cum_pnl_usd = 0.0
        trades_history = []
        last_long_cond, last_short_cond = False, False

        for i in range(50, len(df) - 1):
            curr, prev, nxt = df.iloc[i], df.iloc[i-1], df.iloc[i+1]
    
      
                # 第一層邏輯：mode 1 出場判斷
            if pos != 0 and i > entry_idx: 
                    exited, pnl_pct, out_p, out_t = False, 0.0, 0.0, nxt['time']
                    tp_p = entry_p * (1 + self.TP_PCT/100) if pos == 1 else entry_p * (1 - self.TP_PCT/100)
                    sl_p = entry_p * (1 - self.SL_PCT/100) if pos == 1 else entry_p * (1 + self.SL_PCT/100)

                    
                    #立即止盈止損
                    if (pos == 1 and curr['high'] >= tp_p) or (pos == -1 and curr['low'] <= tp_p):
                        pnl_pct, exited, out_p, out_t = self.TP_PCT, True, tp_p, nxt['time']
                    elif (pos == 1 and curr['low'] <= sl_p) or (pos == -1 and curr['low'] >= sl_p):
                        pnl_pct, exited, out_p, out_t = -self.SL_PCT, True, sl_p, nxt['time']
                    

                    #形態止盈止損
                    if not exited:
                        curr_profit_pts = (nxt['close'] - entry_p) if pos == 1 else (entry_p - nxt['close'])

                        safe_exit =( abs(curr['close'] - entry_p) >= self.MIN_MOVE_PTS) and curr_profit_pts > 0

                        safe_loss_exit =( abs(curr['close'] - entry_p) >= self.MAX_LOSS_PTS )and curr_profit_pts < 0
                     
                       
                        morph_sig = (pos == 1 and  curr['loss_l'])or   (pos == -1 and curr['loss_s'])


                        move_pct = ((nxt['close'] - entry_p)/entry_p)*100 if pos==1 else ((entry_p - nxt['close'])/entry_p)*100


                        if (safe_exit and morph_sig) or( safe_loss_exit and morph_sig) :
                            pnl_pct, exited, out_p, out_t = move_pct, True, nxt['close'], nxt['time']

                    if exited:
                        net_pnl = (self.TRADE_QTY_USD * (pnl_pct/100)) - (self.TRADE_QTY_USD * self.COMMISSION_PCT * 2)
                        cum_pnl_usd += net_pnl
                     
                        trades_history.append({
                            "no": len(trades_history) + 1, "side": "看多" if pos==1 else "看空",
                            "in_t": entry_t, "out_t": out_t, "in_p": entry_p, "out_p": out_p,
                            "net": net_pnl, "cum": cum_pnl_usd
                        })
                        pos, entry_idx = 0, -1
                        self.last_exit_bar_time = curr['time']  # <--- 新增這行
                        continue


             #2.進場邏輯===================                                                                                                                                                                                                                                                                                                                                                   # --- 2. 進場邏輯 ---
            if pos == 0:
                    if self.last_exit_bar_time is not None and curr['time'] == self.last_exit_bar_time:
                        continue  # 本K棒已經出場過，不再進場
                    # 判斷 Breakout (加上效率比過濾)
                    is_climax = curr['body'] > curr['avg_body'] * self.CLIMAX_MULT
                    is_gs = curr['close'] > curr['open'] and prev['close'] > prev['open']
                    is_rs = curr['close'] < curr['open'] and prev['close'] < prev['open']
                
                    is_bk = ((curr['close'] > curr['upper15'] and (is_climax or is_gs)) or \
                            (curr['close'] < curr['lower15'] and (is_climax or is_rs))) and curr['isEfficient']
                
                    # 判斷窄/寬區間
                    is_nr = curr['adx_v'] >= self.ADX_THRESHOLD and curr['bbw_v'] <= self.BBW_THRESHOLD and not is_bk
                    is_wd = curr['adx_v'] >= 20 and curr['bbw_v'] > self.BBW_THRESHOLD and not is_bk
                
                    # 總進場許可 (加入 R2 與 效率比)
                    can_enter = (is_bk or is_nr or (is_wd and curr['wide_vol_ok'])) and curr['isEfficient'] and curr['is_linear_trend']

                    last_long_cond = (can_enter and curr['close'] > curr['upper15'] and curr['close'] > curr['m5_hh'])
                    last_short_cond = (can_enter and curr['close'] < curr['lower15'] and curr['close'] < curr['m5_ll'])

                    if last_long_cond:
                  
                            pos, entry_p, entry_t, entry_idx = 1, nxt['open'], nxt['time'], i + 1
                          


                    elif last_short_cond:
                   
                             pos, entry_p, entry_t, entry_idx = -1, nxt['open'], nxt['time'], i + 1
                             

        # 返回歷史清單與當前狀態
        return trades_history, {
            "pos": pos, "entry_p": entry_p, "cum_pnl": cum_pnl_usd,
            "long_cond": last_long_cond, "short_cond": last_short_cond,
            "df": df
        }

    def display_trade_history(self, trades):
        """🚀 封裝後的交易記錄顯示函數"""
        print("\n" + "="*110)
        print(f"{'交易 #':<6} {'類型':<6} {'進場時間':<15} {'出場時間':<15} {'進場價':<9} {'出場價':<9} {'單筆淨利':<12} {'累積損益'}")
        print("-" * 110)
        for t in trades:
            color = Fore.GREEN if t['net'] > 0 else Fore.RED
            print(f"{t['no']:<7} {t['side']:<6} {t['in_t'].strftime('%m/%d %H:%M'):<15} {t['out_t'].strftime('%m/%d %H:%M'):<15} {t['in_p']:<9.1f} {t['out_p']:<9.1f} {color}{t['net']:+.3f} USD{Style.RESET_ALL:<5} {t['cum']:.3f}")

    def update_monitor_panel(self, state):
        """📊 實時更新的監控面板 (移除 ||)"""
        last_bar = state['df'].iloc[-1]
        pos, entry_p = state['pos'], state['entry_p']
        
        # --- 新增：計算目前時間與 K 線倒數 ---
        now = datetime.now()
        current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # 5 分鐘線 (300秒) 的倒數邏輯
        seconds_passed = (now.minute % 5) * 60 + now.second
        remaining_seconds = 300 - seconds_passed
        m, s = divmod(remaining_seconds, 60)
        countdown_str = f"{m:02d}分{s:02d}秒"


                                                                                                                                                                                                                                                                                                                                                                # 訊號判定
        if pos == 1: status_text, status_color = "持有多單", Fore.CYAN
        elif pos == -1: status_text, status_color = "持有空單", Fore.MAGENTA
        elif state['long_cond']: status_text, status_color = " 建議做多", Fore.GREEN
        elif state['short_cond']: status_text, status_color = " 建議做空", Fore.RED
        else: status_text, status_color = "等待趨勢", Fore.WHITE

        tp_price = entry_p * (1 + self.TP_PCT/100) if pos == 1 else entry_p * (1 - self.TP_PCT/100) if pos == -1 else None
        sl_price = entry_p * (1 - self.SL_PCT/100) if pos == 1 else entry_p * (1 + self.SL_PCT/100) if pos == -1 else None
        
        exit_trigger = (pos == 1 and (last_bar['is_rev_r'] or last_bar['loss_l'])) or \
                        (pos == -1 and (last_bar['is_rev_g'] or last_bar['loss_s']))
        
        cur_q = self.TRADE_QTY_USD / entry_p if pos != 0 else 0
        p_usd = 0.0
        if pos == 1: p_usd = (last_bar['close'] - entry_p) * cur_q - (entry_p * cur_q * self.COMMISSION_PCT * 2)
        elif pos == -1: p_usd = (entry_p - last_bar['close']) * cur_q - (entry_p * cur_q * self.COMMISSION_PCT * 2)

        print("\n" + "╔" + "═"*45 + "╗")
        print(f"  {'📊 BTC 形態識別 1 - 即時監控面板':<35}  ")
        print("╠" + "═"*45 + "╣")
        # 新增兩行：目前時間 與 倒數計時
        print(f"  目前時間: {Fore.WHITE}{current_time_str:<33}{Style.RESET_ALL}  ")
        print(f"  下一根 K 線倒數: {Fore.YELLOW}{countdown_str:<30}{Style.RESET_ALL}  ")
        print("-" * 47)
        print(f"  訊號狀態: {status_color}{status_text:<33}{Style.RESET_ALL}  ")
        print(f"  止盈價格: {Fore.GREEN}{f'{tp_price:.1f}' if tp_price else '--':<33}{Style.RESET_ALL}  ")
        print(f"  止損價格: {Fore.RED}{f'{sl_price:.1f}' if sl_price else '--':<33}{Style.RESET_ALL}  ")
        
        morph_color = Fore.RED if exit_trigger else Fore.WHITE
        morph_price = f"{last_bar['open']:.1f}" if (pos != 0 and exit_trigger) else "--"
        print(f"  形態平倉價格: {morph_color}{morph_price:<29}{Style.RESET_ALL}  ")
        
        pnl_color = Fore.GREEN if p_usd >= 0 else Fore.RED
        print(f"  當前獲利: {pnl_color}{f'{p_usd:.2f} USD':<33}{Style.RESET_ALL}  ")
        print(f"  BTC 目前價格: {Fore.YELLOW}{f'{last_bar['close']:.1f}':<31}{Style.RESET_ALL}  ")
        
        entry_color = Fore.BLUE
        entry_display = f"{entry_p:.1f}" if pos != 0 else (f"{last_bar['upper15']:.1f}" if state['long_cond'] else f"{last_bar['lower15']:.1f}")
        print(f"  進場價格: {entry_color}{entry_display:<33}{Style.RESET_ALL}  ")
        print("╚" + "═"*45 + "╝")


        
                                                                                                                                                                                                                                                                                                                                                                                                                    # --- 新增下單同步函數 ---
    def sync_demo_orders(self, current_state):
        """根據核心邏輯狀態與雲端實時倉位進行對接"""
        try:

            #----------------------------------------------------------------------------
            # --- 1. 抓取【正式盤】即時價格 (裁判) ---
            ticker = self.data_exchange.fetch_ticker(self.SYMBOL)
            if ticker is None or 'last' not in ticker: 
                return # 預防 NoneType 錯誤
            
            real_price = ticker['last'] # 這是全世界公認的真實價格
            #----------------------------------------------------------------------------

           # --- 2. 抓取【模擬盤】持倉狀態 (執行環境) ---
            pos_info = self.exchange.fetch_positions([self.SYMBOL])
            real_amt = 0.0
            target_symbol = self.SYMBOL.replace('/', '')
            for p in pos_info:
               
                p_symbol = p.get('info', {}).get('symbol', '')
                if p_symbol == target_symbol:
                
                    real_amt = float(p.get('contracts', 0) or 0)
            
                                                                                                                                                                                                                                                                                                                                                                                                                                                        # 2. 獲取邏輯應持有的狀態
            logic_pos = current_state['pos']



         
            actual_qty = 0.0    # 實際合約數量 (例如 0.001)
            display_side = "空倉"

            target_symbol = self.SYMBOL.replace('/', '')

             # --- 2. 抓取【實盤/模擬盤】持倉狀態 ---
            pos_info = self.exchange.fetch_positions([self.SYMBOL])
            real_amt = 0        # 用於邏輯判定 (1, 0, -1)
            actual_qty = 0.0    # 實際合約數量 (例如 0.001)
            display_side = "空倉"

            target_symbol = self.SYMBOL.replace('/', '')

            for p in pos_info:
                # 幣安返回的 symbol 通常不帶斜槓
                if p.get('symbol') == target_symbol or p.get('info', {}).get('symbol') == target_symbol:
                    contracts = float(p.get('contracts', 0) or 0)
        
                    if contracts > 0:
                        actual_qty = contracts
                        side = p.get('side')  # 'long' 或 'short'
            
                        # 轉換為你的邏輯代碼
                        real_amt = 1 if side == 'long' else -1
                        display_side = "做多 🟢" if side == 'long' else "做空 🔴"

            # ==========================================
            # 📊 在終端機打印目前狀態 (Debug 專用)
            # ==========================================
            print("-" * 50)
            print(f"{Fore.CYAN}【帳戶狀態檢查】")
            print(f"當前標的: {self.SYMBOL}")
            print(f"雲端持倉: {Fore.WHITE}{display_side}")
            print(f"實際數量: {actual_qty}")
            print(f"判定代碼 (real_amt): {Fore.MAGENTA}{real_amt}")
            print(f"策略訊號 (logic_pos): {Fore.MAGENTA}{logic_pos}")
            print("-" * 50)
            
            # 邏輯 A: 應持多單 (1) 但雲端無部位或持空單
            if (logic_pos == 1 and real_amt == 0):
                qty = self.exchange.amount_to_precision(self.SYMBOL, self.TRADE_QTY_USD / current_state['df'].iloc[-1]['close'])
                print("=" *50)
                print(f"{Fore.YELLOW}進場：執行   多單買入...")
                print("=" *50)
                self.exchange.create_market_order(self.SYMBOL, 'buy', qty)
                time.sleep(10)
            
            # 邏輯 B: 應持空單 (-1) 但雲端無部位或持多單
            elif (logic_pos == -1 and real_amt == 0):
                qty = self.exchange.amount_to_precision(self.SYMBOL, self.TRADE_QTY_USD / current_state['df'].iloc[-1]['close'])
                print("=" *50)
                print(f"{Fore.YELLOW}進場：執行   空單賣出...")
                print("=" *50)
                self.exchange.create_market_order(self.SYMBOL, 'sell', qty)
                time.sleep(10)
                
            # 邏輯 C: 應空倉 (0) 但雲端仍有部位 (做空)
            elif logic_pos == 0 and real_amt == -1:
                qty = self.exchange.amount_to_precision(self.SYMBOL, self.TRADE_QTY_USD / current_state['df'].iloc[-1]['close'])
                print("=" *50)
                print(f"{Fore.YELLOW}進場：執行  做空平倉...")
                print("=" *50)
                self.exchange.create_market_order(self.SYMBOL, 'buy', qty)
                time.sleep(10)
            # 邏輯 D: 應空倉 (0) 但雲端仍有部位 (做多)
            elif  logic_pos == 0 and real_amt == 1:
                qty = self.exchange.amount_to_precision(self.SYMBOL, self.TRADE_QTY_USD / current_state['df'].iloc[-1]['close'])
                print("=" *50)
                print(f"{Fore.YELLOW}進場：執行  做多平倉...")
                print("=" *50)
                self.exchange.create_market_order(self.SYMBOL, 'sell', qty)
                time.sleep(10)
            #應持多單(1)但持有空單
            elif  (logic_pos == 1 and real_amt == -1):
               
                qty = self.exchange.amount_to_precision(self.SYMBOL, self.TRADE_QTY_USD / current_state['df'].iloc[-1]['close'])
                print("=" *50)
                print(f"{Fore.YELLOW}進場：執行   空單平倉 接著做多...")
                print("=" *50)
                self.exchange.create_market_order(self.SYMBOL, 'buy', qty)
                time.sleep(10)
            #應持空單(-1) 但持有多單
            elif ( logic_pos == -1 and real_amt == 1):

                qty = self.exchange.amount_to_precision(self.SYMBOL, self.TRADE_QTY_USD / current_state['df'].iloc[-1]['close'])
                print("=" *50)
                print(f"{Fore.YELLOW}進場：執行   多單平倉 接著做空...")
                print("=" *50)
                self.exchange.create_market_order(self.SYMBOL, 'sell', qty)
                time.sleep(10)



        except Exception as e:
            print(f"⚠️ 下單同步失敗: {e}")

if __name__ == "__main__":
    # --- 在此填入你的 Demo Key 即可執行下單 ---
    DEMO_API = "xng5xOOaHn9KROZDnWP57PYFxY1LMvWjR0Z7DLt8R4u2jwP6mOCj6HDRUyCVNLpM"
    DEMO_SECRET = "g3GCHjxKaowQJ0qGxgF5QUC2AW0wbWcobthLk8rdvPaLgj8b3l7cxBC1hdmLBiTU"
    backtester = BTCBacktesterUTC8(DEMO_API, DEMO_SECRET)
    
    '''
    df = backtester.fetch_data("2025-12-08 17:00")
    trades, current_state = backtester.run_simulation(df)
    backtester.display_trade_history(trades)
    '''
   

    
    while True:
        try:
            # 1. 抓取最新數據
            df = backtester.fetch_data("2025-01-05 17:00")
            # 2. 重新計算狀態
            trades, current_state = backtester.run_simulation(df)
            # 3. 實時顯示面板
            backtester.update_monitor_panel(current_state)
            
            # --- 實時下單同步 ---
            
            if DEMO_API:
                backtester.sync_demo_orders(current_state)
                
            # 4. 每 10 秒刷新一次
            time.sleep(10)
            
        except Exception as e:
            print(f"❌ 更新錯誤: {e}，5秒後重試...")
            time.sleep(5)
       
  
