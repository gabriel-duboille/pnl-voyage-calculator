# P&L Voyage Calculator

This project was built for educational purposes by a 2nd-year Accounting & Finance student at the London School of Economics. The goal was to understand how physical commodity trading firms use transformation in space: buying a commodity at one location and selling it at another — to generate profits, and how they manage the financial risks involved (hedging, freight exposure, financing costs, carbon liabilities).

The tool is not intended for live trading. It is a simulation environment that connects real market data to simplified logistics and risk models.

## What It Does

- **Market Intelligence**: Live spot prices, real futures strip forward curves (CME/NYMEX/CBOT/COMEX via Yahoo Finance), 10-year seasonality analysis, and cross-commodity correlations across multiple time horizons.
- **Voyage Modeling**: Route distances, steaming times, and canal transit fees (Suez, Panama) based on a global port and chokepoint database.
- **Vessel Integration**: Specs for VLCC, Suezmax, Aframax, MR Tanker, LNG Carrier, Capesize, and Panamax — including fuel burn, capacity, and dimensions.
- **Hedging**: Configurable hedge ratio with futures lock-in price. P&L splits into locked margin vs. unhedged exposure.
- **P&L Waterfall**: Full cost breakdown including freight, fuel, canal fees, EU ETS carbon (with the 50%/100% intra-EU rule), insurance, SOFR-based financing, and demurrage.
- **Risk Analysis**: Sensitivity heatmaps, 3D risk surface, and Monte Carlo VaR & CVaR using a Student-t distribution calibrated to historical tail risk.
- **Contango Carry Scanner**: Scans all commodities and routes for profitable carry trades by comparing spot prices against voyage-duration-matched forward contracts, net of all costs.

## Tech Stack

- **Python 3.9+** / **Streamlit**
- **YFinance**: live commodity prices, futures strips, Treasury rates, carbon proxies
- **Plotly**: financial charts, waterfall, heatmaps, 3D surfaces
- **Folium**: interactive route maps
- **SciPy**: fat-tailed Monte Carlo (Student-t VaR/CVaR)

## Development

The author defined the functionalities, quantitative logic, and UI structure. The code was largely generated and iterated using AI assistance (Claude, Anthropic). Any bugs or logical gaps are the author's responsibility.

## Disclaimer

This is a simulation tool for educational purposes. It does not constitute financial or operational advice for live trading.

## Quick Start

1. **Market Intelligence**: Select a commodity to load spot data, forward curve, seasonality, and correlations.
2. **Trade Configuration**: Set ports, vessel, pricing, hedge parameters, and costs.
3. **P&L Results**: Review the waterfall, run sensitivity analysis, and check VaR/CVaR.
4. **Contango Carry Scanner**: Scan for profitable carry trades across all assets and routes.
