# P&L Voyage Calculator

The P&L Voyage Calculator is a decision-support tool designed to simulate the financial viability and operational risks of commodity trading voyages. It integrates real-time market data with logistical modeling to provide an end-to-end overview of trade profitability.

## Core Functionalities

* **Market Analysis**: Real-time spot price tracking, momentum analysis, and implied forward curve projections for energy, metals, and agricultural commodities.
* **Voyage Modeling**: Logistical calculation of distances and steaming times based on a global port and chokepoint database.
* **Vessel Technical Integration**: Detailed specification modeling for VLCC, Suezmax, Aframax, LNG, and Dry Bulk carriers, including fuel burn rates and cargo capacities.
* **Financial Risk Management**: Waterfall P&L analysis incorporating SOFR-based financing, insurance premiums, demurrage rates, and EU ETS carbon liabilities.
* **Sensitivity & Stress Testing**: Interactive heatmaps and 3D risk topography evaluating the impact of price volatility versus freight cost variations on net profit.
* **Predictive Modeling**: Monte Carlo simulations to calculate 95% Value at Risk (VaR) and Expected Mean profit based on past monthly asset drift.

## Technical Stack

* **Language**: Python 3.9+
* **Framework**: Streamlit
* **Data Sourcing**: YFinance API (Live feeds for Commodities, Heating Oil proxy for Fuel, and EUA Carbon)
* **Visualization**: Plotly (Financial Charts) and Folium (Geospatial Mapping)
* **Computation**: NumPy and Pandas

## Development & Attribution

This project was developed through a hybrid approach combining academic foundations with generative AI implementation.

* **Logical Framework**: The selection of functionalities, quantitative risk parameters, UI design, and chart types was directed entirely by the author.
* **Background**: At the time of development, the author possessed foundational experience in Python through Programming modules at the London School of Economics (ST101, ST115).
* **Technical Implementation**: While the author defined the architectural requirements and logical flow, the majority of the code was generated and refined using AI assistance.
* **Responsibility**: Any remaining bugs or logical inconsistencies are the sole responsibility of the author.

## Disclaimer

This software is for simulation purposes only. It does not constitute financial advice or operational guidance for live trading environments.

## Instructions for Use

1. **Select Commodity**: Choose an asset to initialize market analysis and fetch real-time spot prices.
2. **Configure Voyage**: Set Loading/Discharge ports and select an appropriate vessel class for the cargo.
3. **Analyze P&L**: Review the financial waterfall and run stress tests to evaluate trade risk.
