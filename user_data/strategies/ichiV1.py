import logging
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (CategoricalParameter, DecimalParameter, 
                                IntParameter)
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'
import technical.indicators as ftt
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy import stoploss_from_open
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

class ichiV1(IStrategy):

    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v0.0.001"

    # Futures
    custom_leverage = 1.0

    # NOTE: settings as of the 25th july 21
    # Buy hyperspace params:
        #"buy_min_fan_magnitude_gain": 1.002 # NOTE: Good value (Win% ~70%), alot of trades
        #"buy_min_fan_magnitude_gain": 1.008 # NOTE: Very save value (Win% ~90%), only the biggest moves 1.008,

    buy_trend_above_senkou_level = IntParameter(1,8, default=5, space="buy")
    buy_trend_bullish_level = IntParameter(1,8, default=6, space="buy")
    buy_fan_magnitude_shift_value = IntParameter(1,8, default=5, space="buy")
    buy_min_fan_magnitude_gain = DecimalParameter(0.980, 1.020, default=1.002, space="buy")
    
    # Sell hyperspace params:
    # NOTE: was 15m but kept bailing out in dryrun
    sell_trend_indicator = CategoricalParameter(["trend_close_5m", "trend_close_15m", "trend_close_30m", "trend_close_1h", "trend_close_2h", "trend_close_4h", "trend_close_6h", "trend_close_8h"], default="trend_close_15m", space="sell")

    # ROI table:
    minimal_roi = {
        "0": 0.061 * custom_leverage,
        "3": 0.023 * custom_leverage,
        "6": 0.008 * custom_leverage,
        "19": 0 * custom_leverage
    }

    # Stoploss:
    stoploss = -0.217 * custom_leverage

    # Optimal timeframe for the strategy
    timeframe = '1m'

    startup_candle_count = 130
    #startup_candle_count = 96
    #process_only_new_candles = True
    process_only_new_candles = False

    trailing_stop = False
    #trailing_stop_positive = 0.002
    #trailing_stop_positive_offset = 0.025
    #trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    #calcola solo gli indicatori strettamente necessari se True
    optimize = False
    
    plot_config = {
        'main_plot': {
            # fill area between senkou_a and senkou_b
            'senkou_a': {
                'color': 'green', #optional
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud', #optional
                'fill_color': 'rgba(255,76,46,0.2)', #optional
            },
            # plot senkou_b, too. Not only the area to it.
            'senkou_b': {},
            'trend_close_5m': {'color': '#FF5733'},
            'trend_close_15m': {'color': '#FF8333'},
            'trend_close_30m': {'color': '#FFB533'},
            'trend_close_1h': {'color': '#FFE633'},
            'trend_close_2h': {'color': '#E3FF33'},
            'trend_close_4h': {'color': '#C4FF33'},
            'trend_close_6h': {'color': '#61FF33'},
            'trend_close_8h': {'color': '#33FF7D'}
        },
        'subplots': {
            'fan_magnitude': {
                'fan_magnitude': {}
            },
            'fan_magnitude_gain': {
                'fan_magnitude_gain': {}
            }
        }
    }
    
    def bot_start(self, **kwargs) -> None:
        """
        Called only once after bot instantiation.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        
        self.optimize = self.config['runmode'].value in ('live', 'dry_run')        
        
        self.custom_leverage = self.config['custom_leverage']
        
        logger.info(f"--> runmode: {self.config['runmode'].value} | custom_leverage: {self.custom_leverage} ")


    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        
        #logger.info(" bot_loop_start ")


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        logger.info(f"START populate_indicators : {metadata['pair']}")
        
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe_open = heikinashi['open']
        #dataframe_open = dataframe['open']
        #dataframe_close = heikinashi['close']
        dataframe_close = dataframe['close']
        #dataframe['hk_close'] = heikinashi['close']
        #dataframe['hk_high'] = heikinashi['high']
        #dataframe['hk_low'] = heikinashi['low']

        moltiplicatore = 1 #dipende dal timeframe
        
        if moltiplicatore > 1:
            dataframe['trend_close_5m'] = ta.EMA(dataframe_close, timeperiod=1*moltiplicatore)
        else:
            dataframe['trend_close_5m'] = dataframe_close
        
        if not self.optimize or (self.buy_trend_above_senkou_level.value >= 2 or self.buy_trend_bullish_level.value >= 2):
            dataframe['trend_close_15m'] = ta.EMA(dataframe_close, timeperiod=3*moltiplicatore)
        if not self.optimize or (self.buy_trend_above_senkou_level.value >= 3 or self.buy_trend_bullish_level.value >= 3):
            dataframe['trend_close_30m'] = ta.EMA(dataframe_close, timeperiod=6*moltiplicatore)
        #if not self.optimize or (self.buy_trend_above_senkou_level.value >= 4 or self.buy_trend_bullish_level.value >= 4):
        dataframe['trend_close_1h'] = ta.EMA(dataframe_close, timeperiod=12*moltiplicatore)
        #if not self.optimize or (self.buy_trend_above_senkou_level.value >= 5 or self.buy_trend_bullish_level.value >= 5):
        dataframe['trend_close_2h'] = ta.EMA(dataframe_close, timeperiod=24*moltiplicatore)
        if not self.optimize or (self.buy_trend_above_senkou_level.value >= 6 or self.buy_trend_bullish_level.value >= 6):
            dataframe['trend_close_4h'] = ta.EMA(dataframe_close, timeperiod=48*moltiplicatore)
        if not self.optimize or (self.buy_trend_above_senkou_level.value >= 7 or self.buy_trend_bullish_level.value >= 7):
            dataframe['trend_close_6h'] = ta.EMA(dataframe_close, timeperiod=72*moltiplicatore)
        #if not self.optimize or (self.buy_trend_above_senkou_level.value >= 8 or self.buy_trend_bullish_level.value >= 8):
        dataframe['trend_close_8h'] = ta.EMA(dataframe_close, timeperiod=96*moltiplicatore)


        if not self.optimize or (self.buy_trend_bullish_level.value >= 1):
            if moltiplicatore > 1:
                dataframe['trend_open_5m'] = ta.EMA(dataframe_open, timepriod=1*moltiplicatore)
            else:
                dataframe['trend_open_5m'] = dataframe_open
        if not self.optimize or (self.buy_trend_bullish_level.value >= 2):
            dataframe['trend_open_15m'] = ta.EMA(dataframe_open, timeperiod=3*moltiplicatore)
        if not self.optimize or (self.buy_trend_bullish_level.value >= 3):
            dataframe['trend_open_30m'] = ta.EMA(dataframe_open, timeperiod=6*moltiplicatore)
        if not self.optimize or (self.buy_trend_bullish_level.value >= 4):
            dataframe['trend_open_1h'] = ta.EMA(dataframe_open, timeperiod=12*moltiplicatore)
        if not self.optimize or (self.buy_trend_bullish_level.value >= 5):
            dataframe['trend_open_2h'] = ta.EMA(dataframe_open, timeperiod=24*moltiplicatore)
        if not self.optimize or (self.buy_trend_bullish_level.value >= 6):
            dataframe['trend_open_4h'] = ta.EMA(dataframe_open, timeperiod=48*moltiplicatore)
        if not self.optimize or (self.buy_trend_bullish_level.value >= 7):
            dataframe['trend_open_6h'] = ta.EMA(dataframe_open, timeperiod=72*moltiplicatore)
        if not self.optimize or (self.buy_trend_bullish_level.value >= 8):
            dataframe['trend_open_8h'] = ta.EMA(dataframe_open, timeperiod=96*moltiplicatore)

        dataframe['fan_magnitude'] = (dataframe['trend_close_1h'] / dataframe['trend_close_8h'])
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        #dataframe['chikou_span'] = ichimoku['chikou_span'] NON UTILIZZARE: ha bias
        
        #dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        #dataframe['kijun_sen'] = ichimoku['kijun_sen']
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        #dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        #dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        #dataframe['cloud_green'] = ichimoku['cloud_green']
        #dataframe['cloud_red'] = ichimoku['cloud_red']

        #dataframe['atr'] = ta.ATR(dataframe)
        
        logger.info(f"END   populate_indicators : {metadata['pair']}")

        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        logger.info(f"START populate_entry_trend : {metadata['pair']}")
        
        conditions = []

        # Trending market
        if self.buy_trend_above_senkou_level.value >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 2:
            conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 3:
            conditions.append(dataframe['trend_close_30m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_30m'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 4:
            conditions.append(dataframe['trend_close_1h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_1h'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 5:
            conditions.append(dataframe['trend_close_2h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_2h'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 6:
            conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 7:
            conditions.append(dataframe['trend_close_6h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_6h'] > dataframe['senkou_b'])

        if self.buy_trend_above_senkou_level.value >= 8:
            conditions.append(dataframe['trend_close_8h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_8h'] > dataframe['senkou_b'])

        # Trends bullish
        if self.buy_trend_bullish_level.value >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['trend_open_5m'])

        if self.buy_trend_bullish_level.value >= 2:
            conditions.append(dataframe['trend_close_15m'] > dataframe['trend_open_15m'])

        if self.buy_trend_bullish_level.value >= 3:
            conditions.append(dataframe['trend_close_30m'] > dataframe['trend_open_30m'])

        if self.buy_trend_bullish_level.value >= 4:
            conditions.append(dataframe['trend_close_1h'] > dataframe['trend_open_1h'])

        if self.buy_trend_bullish_level.value >= 5:
            conditions.append(dataframe['trend_close_2h'] > dataframe['trend_open_2h'])

        if self.buy_trend_bullish_level.value >= 6:
            conditions.append(dataframe['trend_close_4h'] > dataframe['trend_open_4h'])

        if self.buy_trend_bullish_level.value >= 7:
            conditions.append(dataframe['trend_close_6h'] > dataframe['trend_open_6h'])

        if self.buy_trend_bullish_level.value >= 8:
            conditions.append(dataframe['trend_close_8h'] > dataframe['trend_open_8h'])

        # Trends magnitude
        conditions.append(dataframe['fan_magnitude_gain'] >= self.buy_min_fan_magnitude_gain.value)
        conditions.append(dataframe['fan_magnitude'] > 1)

        for x in range(self.buy_fan_magnitude_shift_value.value):
            conditions.append(dataframe['fan_magnitude'].shift(x+1) < dataframe['fan_magnitude'])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        logger.info(f"END   populate_entry_trend : {metadata['pair']}")

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        conditions.append(qtpylib.crossed_below(dataframe['trend_close_5m'], dataframe[self.sell_trend_indicator.value]))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return self.custom_leverage
