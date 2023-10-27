
list-exchanges:
	docker compose run --rm freqtrade list-exchanges

list-timeframes:
	docker compose run --rm freqtrade list-timeframes -c user_data/exchange-binance.json

list-markets:
	docker compose run --rm freqtrade list-markets -c user_data/exchange-binance.json --print-json

data:
	docker compose run --rm freqtrade download-data --timeframe 1m 3m 5m -c user_data/exchange-binance.json --timerange 20221201-20231001

clean:
	rm -rf user_data/models/*
	rm -rf user_data/backtest_results/*

NOW=$(shell date +%Y%m%d%H%M)
#TIMEFRAME=15m
#TIMEFRAME=5m
TIMEFRAME=1m
TIMERANGE=20230801-20230901
#TIMEFRAMEDETAIL=--timeframe-detail 1m
TIMEFRAMEDETAIL=
CURRENCY=

backtesting:
	docker compose run --rm freqtrade backtesting --strategy-list ichiV1 --timeframe $(TIMEFRAME) $(TIMEFRAMEDETAIL) -c user_data/config.json -c user_data/exchange-binance$(CURRENCY).json  --timerange $(TIMERANGE) --breakdown day week month > $(TIMERANGE)-$(TIMEFRAME)-results.txt && cat $(TIMERANGE)-$(TIMEFRAME)-results.txt 

plot:
	docker compose run --rm freqtrade plot-dataframe --strategy ichiV1 --timeframe $(TIMEFRAME) -c user_data/config.json -c user_data/exchange-binance$(CURRENCY).json -c user_data/config-ai.json -c user_data/config-ai-$(TIMEFRAME).json --timerange $(TIMERANGE) 

plot-profit:
	docker compose run --rm freqtrade plot-profit --strategy ichiV1 --timeframe $(TIMEFRAME) -c user_data/config.json -c user_data/exchange-binance$(CURRENCY).json -c user_data/config-ai.json -c user_data/config-ai-$(TIMEFRAME).json --timerange $(TIMERANGE) 

generate-results: backtesting plot plot-profit
	mkdir user_data/results/$(NOW)-$(TIMERANGE)-$(TIMEFRAME)$(CURRENCY)/
	mkdir user_data/results/$(NOW)-$(TIMERANGE)-$(TIMEFRAME)$(CURRENCY)/models
	mv $(TIMERANGE)-$(TIMEFRAME)-results.txt user_data/results/$(NOW)-$(TIMERANGE)-$(TIMEFRAME)$(CURRENCY)/$(TIMERANGE)-$(TIMEFRAME)-results.txt
	mv user_data/plot/freqtrade-plot-* user_data/results/$(NOW)-$(TIMERANGE)-$(TIMEFRAME)$(CURRENCY)/$(TIMERANGE)-$(TIMEFRAME)-plot.html
	mv user_data/plot/freqtrade-profit-plot.html user_data/results/$(NOW)-$(TIMERANGE)-$(TIMEFRAME)$(CURRENCY)/$(TIMERANGE)-$(TIMEFRAME)-plot-profit.html
	cp user_data/strategies/* user_data/results/$(NOW)-$(TIMERANGE)-$(TIMEFRAME)$(CURRENCY)/
	cp user_data/config*.json user_data/results/$(NOW)-$(TIMERANGE)-$(TIMEFRAME)$(CURRENCY)/
	cp user_data/exchange-*.json user_data/results/$(NOW)-$(TIMERANGE)-$(TIMEFRAME)$(CURRENCY)/


generate-results-5m: TIMEFRAME=5m
generate-results-5m: generate-results

generate-results-3m: TIMEFRAME=3m
generate-results-3m: generate-results

## BTC Only

generate-results-btc: CURRENCY=-btc
generate-results-btc: generate-results

generate-results-btc-3m: TIMEFRAME=3m
generate-results-btc-3m: generate-results-btc

generate-results-btc-5m: TIMEFRAME=5m
generate-results-btc-5m: generate-results-btc

## SOL Only

generate-results-sol: CURRENCY=-sol
generate-results-sol: generate-results

generate-results-sol-3m: TIMEFRAME=3m
generate-results-sol-3m: generate-results-sol

generate-results-sol-5m: TIMEFRAME=5m
generate-results-sol-5m: generate-results-sol

