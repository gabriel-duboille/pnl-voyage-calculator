import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
import textwrap
from streamlit_folium import st_folium

# Imports from renamed modules
from data_engine_pnl_voyage_calculator import (
    COMMODITIES, get_market_data, get_correlations, 
    get_macro_data, get_carbon_price, get_fuel_market_price
)
from logic_engine_pnl_voyage_calculator import (
    PORT_COORDINATES, CHOKEPOINTS, ROUTES_DB, SHIPS, COMMODITY_SPECS,
    get_route_data, calculate_voyage_metrics, calculate_carbon_cost, calculate_final_pnl,
    calculate_monte_carlo_var, calculate_sensitivity_grid
)

# --- Configuration & CSS ---
st.set_page_config(page_title="P&L Voyage Calculator", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stMarkdown, .stHeader, .stSubheader, h1, h2, h3, h4, p, span, div { color: #FAFAFA !important; }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] { 
        background-color: #1c1f26; color: #FAFAFA; border-color: #444; 
    }
    div[data-testid="stMetric"] { 
        background-color: #161920; border: 1px solid #333; padding: 15px; border-radius: 8px; 
    }
    div[data-testid="stMetricLabel"] { color: #aaa !important; }
    div[data-testid="stMetricValue"] { color: #FAFAFA !important; }
    div.stButton > button { 
        width: 100%; background-color: #0E1117; color: #FAFAFA; 
        border: 1px solid #30333F; border-radius: 4px; height: 50px; 
        font-size: 16px; text-align: left; padding-left: 15px; margin-bottom: 5px; 
    }
    div.stButton > button:hover { border-color: #3388ff; color: #3388ff; background-color: #161920; }
    .leaflet-control-attribution { display: none !important; }
    html, body, [class*="css"] { font-size: 16px; }
</style>
""", unsafe_allow_html=True)

# --- State Management Helpers ---
def init_state(key, default_value):
    if key not in st.session_state:
        st.session_state[key] = default_value

def save_state(perm_key, widget_key):
    st.session_state[perm_key] = st.session_state[widget_key]

# Initialize permanent memory
init_state('current_view', 'Commodity Selection')
init_state('selected_asset', None)
init_state('p_origin', "Houston (US)")
init_state('p_destination', "Rotterdam (ARA)")
init_state('p_ship_type', "Aframax (Crude/Prods)")
init_state('p_volume', 700000)
init_state('p_freight_rate', 0.0)
init_state('p_buy_price', 0.0)
init_state('p_sell_price', 0.0)
init_state('p_fuel_price', 600.0)
init_state('p_carbon_price', 75.0)
init_state('p_insurance', 0.10)
init_state('p_interest', 5.50)
init_state('p_equity', 15)
init_state('p_demurrage_days', 0)
init_state('p_demurrage_rate', 35000.0)

# Sync shadow keys (widget persistence)
st.session_state['_p_origin'] = st.session_state['p_origin']
st.session_state['_p_destination'] = st.session_state['p_destination']
st.session_state['_p_ship_type'] = st.session_state['p_ship_type']
st.session_state['_p_volume'] = st.session_state['p_volume']
st.session_state['_p_buy_price'] = st.session_state['p_buy_price']
st.session_state['_p_sell_price'] = st.session_state['p_sell_price']
st.session_state['_p_freight_rate'] = st.session_state['p_freight_rate']
st.session_state['_p_fuel_price'] = st.session_state['p_fuel_price']
st.session_state['_p_carbon_price'] = st.session_state['p_carbon_price']
st.session_state['_p_interest'] = st.session_state['p_interest']
st.session_state['_p_insurance'] = st.session_state['p_insurance']
st.session_state['_p_equity'] = st.session_state['p_equity']
st.session_state['_p_demurrage_days'] = st.session_state['p_demurrage_days']
st.session_state['_p_demurrage_rate'] = st.session_state['p_demurrage_rate']

# --- Navigation ---
def set_view(v): 
    st.session_state['current_view'] = v
    if v != "Arbitrage Scanner": st.session_state['show_arb_ribbon'] = False

st.sidebar.title("P&L Voyage Calculator")
st.sidebar.caption("Navigation")
if st.sidebar.button("Commodity Selection"): set_view("Commodity Selection")
if st.sidebar.button("Voyage Configuration"): set_view("Voyage Configuration")
if st.sidebar.button("Profit & Loss Analysis"): set_view("Profit & Loss Analysis")

st.sidebar.markdown("---")
if st.sidebar.button("Arbitrage Scanner"): set_view("Arbitrage Scanner")

st.sidebar.markdown("---")
st.sidebar.caption(f"Active: {st.session_state['current_view']}")

# --- Data Fetching ---
selected_name = st.session_state['selected_asset']
if selected_name:
    selected_ticker = COMMODITIES[selected_name]
    with st.spinner(f"Loading {selected_name} data..."):
        market_data = get_market_data(selected_ticker)
        macro_data = get_macro_data()
        current_price = market_data['price'] if market_data else 0.0
        if st.session_state['p_buy_price'] == 0.0: 
            st.session_state['p_buy_price'] = st.session_state['_p_buy_price'] = float(current_price)
        if st.session_state['p_sell_price'] == 0.0: 
            st.session_state['p_sell_price'] = st.session_state['_p_sell_price'] = float(current_price * 1.05)
else: market_data, macro_data, current_price = None, None, 0.0

# --- UI Helpers ---
def style_correlation_table(df):
    return df.style.background_gradient(cmap='RdYlGn', subset=['Correlation'], vmin=-1, vmax=1).format({"Correlation": "{:.2f}"})

def render_blueprint_card(ship_type, s_data, cargo_type):
    n_loa, n_speed, n_fuel = min(s_data['loa']/350*100, 100), min(s_data['speed']/25*100, 100), min(s_data['fuel_burn']/80*100, 100)
    return f"""
<div style="background-color: #161920; border: 1px solid #333; border-radius: 8px; padding: 15px; font-family: sans-serif; height: 400px; display: flex; flex-direction: column; justify-content: space-between;">
    <div>
        <h4 style="margin: 0; color: #3388ff;">{ship_type}</h4>
        <p style="color: #666; font-size: 12px; margin-top: 5px;">CLASS: {cargo_type.upper()}</p>
        <hr style="border-color: #333; margin: 10px 0;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 13px; color: #ccc;">
            <div>LOA: <span style="color: #fff;">{s_data.get('loa')}m</span></div><div>BEAM: <span style="color: #fff;">{s_data.get('beam')}m</span></div>
            <div>DRAFT: <span style="color: #fff;">{s_data.get('draft')}m</span></div><div>CAP: <span style="color: #fff;">{s_data.get('capacity')/1000:.0f}k</span></div>
        </div>
    </div>
    <div style="margin-top: 20px;">
        <div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; font-size: 12px; color: #888;"><span>SIZE PROFILE</span><span>{s_data.get('loa')}m</span></div><div style="height: 6px; background: #333; border-radius: 3px; overflow: hidden;"><div style="width: {n_loa}%; height: 100%; background: #00CC96;"></div></div></div>
        <div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; font-size: 12px; color: #888;"><span>SPEED</span><span>{s_data.get('speed')} kts</span></div><div style="height: 6px; background: #333; border-radius: 3px; overflow: hidden;"><div style="width: {n_speed}%; height: 100%; background: #3388ff;"></div></div></div>
        <div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; font-size: 12px; color: #888;"><span>FUEL BURN</span><span>{s_data.get('fuel_burn')} mt/d</span></div><div style="height: 6px; background: #333; border-radius: 3px; overflow: hidden;"><div style="width: {n_fuel}%; height: 100%; background: #EF553B;"></div></div></div>
    </div>
</div>"""

# --- V1: COMMODITY SELECTION ---
if st.session_state['current_view'] == "Commodity Selection":
    st.subheader("Asset Selection")
    try: current_idx = list(COMMODITIES.keys()).index(st.session_state['selected_asset'])
    except: current_idx = None
    def update_asset(): st.session_state['p_buy_price'] = st.session_state['p_sell_price'] = 0.0; st.session_state['selected_asset'] = st.session_state['_selected_asset']
    st.selectbox("Select Commodity", list(COMMODITIES.keys()), index=current_idx, key='_selected_asset', on_change=update_asset, placeholder="Choose an asset...", label_visibility="collapsed")
    if not st.session_state['selected_asset']:
        # Removed hand emoji
        st.info("Please select a commodity from the dropdown above to initialize the Market Data."); st.stop()
    if market_data:
        c1, c2, c3 = st.columns(3)
        c1.metric("Last Price", f"${market_data['price']:,.2f}", f"{market_data['delta_1d']*100:.2f}%")
        c2.metric("Weekly Change", f"{market_data['delta_1w']*100:.2f}%"); c3.metric("Monthly Change", f"{market_data['delta_1m']*100:.2f}%")
        st.subheader("Price History"); fig = px.line(market_data['history'], title=f"{selected_name} Trend").update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Market Structure Analysis"); fc1, fc2 = st.columns([2, 1])
        with fc1:
            months = np.arange(1, 7); proj_prices = [current_price * np.exp(market_data['drift_annual'] * (m/12)) for m in months]
            fig_curve = px.line(pd.DataFrame({"Month": [f"M+{m}" for m in months], "Price": proj_prices}), x="Month", y="Price", title="Implied Forward Curve (6M)", markers=True).update_traces(line_color="#00CC96" if market_data['drift_annual'] >= 0 else "#EF553B", line_width=3)
            st.plotly_chart(fig_curve.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)'), use_container_width=True)
            st.caption("Methodology: Projection based on spot price extrapolated using momentum drift (Exponential Model).")
        with fc2:
            st.markdown("##### Structure Details")
            drift_6m = (market_data['drift_annual'] / 2) * 100
            if market_data['drift_annual'] > 0.02: st.info(f"Contango (+{drift_6m:.1f}%)")
            elif market_data['drift_annual'] < -0.02: st.warning(f"Backwardation ({drift_6m:.1f}%)")
            else: st.caption("Neutral Market")
        st.subheader("Asset Correlations"); col_corr1, col_corr2 = st.columns(2)
        with col_corr1: st.write("##### 1 Month Horizon"); c1m = get_correlations(selected_ticker, period="1mo"); st.dataframe(style_correlation_table(c1m.head(5)), use_container_width=True)
        with col_corr2: st.write("##### 1 Week Horizon"); c1w = get_correlations(selected_ticker, period="1wk"); st.dataframe(style_correlation_table(c1w.head(5)), use_container_width=True)

# --- V2: VOYAGE CONFIGURATION ---
elif st.session_state['current_view'] == "Voyage Configuration":
    st.header("Trade Entry & Logistics"); c_map, c_blueprint = st.columns([3, 1])
    ports = sorted(list(set([k[0] for k in ROUTES_DB.keys()] + [k[1] for k in ROUTES_DB.keys()])))
    st.markdown("---"); c_logistics, c_pricing = st.columns(2)
    with c_logistics:
        st.subheader("1. Logistics")
        st.selectbox("Loading Port", ports, index=ports.index(st.session_state.p_origin), key='_p_origin', on_change=save_state, args=('p_origin', '_p_origin'))
        st.selectbox("Discharge Port", ports, index=ports.index(st.session_state.p_destination), key='_p_destination', on_change=save_state, args=('p_destination', '_p_destination'))
        comp_ships = [n for n, s in SHIPS.items() if COMMODITY_SPECS[selected_name] in s["category"]]
        if st.session_state['p_ship_type'] not in comp_ships: st.session_state['p_ship_type'] = comp_ships[0]
        st.selectbox("Vessel Class", comp_ships, index=comp_ships.index(st.session_state.p_ship_type), key='_p_ship_type', on_change=save_state, args=('p_ship_type', '_p_ship_type'))
        st.number_input("Volume (Units)", step=1000, key='_p_volume', on_change=save_state, args=('p_volume', '_p_volume'))
    with c_pricing:
        st.subheader("2. Trade Pricing")
        st.number_input("FOB Price (Buy)", step=0.01, key='_p_buy_price', on_change=save_state, args=('p_buy_price', '_p_buy_price'))
        st.number_input("CIF Price (Sell)", step=0.01, key='_p_sell_price', on_change=save_state, args=('p_sell_price', '_p_sell_price'))
        st.number_input("Charter Rate ($ flat)", step=1000.0, key='_p_freight_rate', on_change=save_state, args=('p_freight_rate', '_p_freight_rate'))
        st.caption("Technical Logic: Set to 0 for vessel class auto-rate. Lump sum values bypass base rate calculations.")
    route_data = get_route_data(st.session_state['p_origin'], st.session_state['p_destination'])
    with c_map:
        st.caption("Live Route Visualization"); wps = route_data['waypoints']; center = wps[len(wps)//2] if wps else [20,0]
        m = folium.Map(location=center, zoom_start=2, tiles="CartoDB dark_matter", attr=' ', attribution_control=False)
        if wps:
            folium.PolyLine(locations=wps, color="#3388ff", weight=3, opacity=0.8, dash_array='5, 10').add_to(m)
            folium.Marker(wps[0], popup="START", icon=folium.Icon(color="green", icon="play")).add_to(m)
            dest_info = PORT_COORDINATES.get(st.session_state['p_destination'], {})
            if dest_info.get("coords"): folium.Marker(dest_info["coords"], popup="END", icon=folium.Icon(color="blue" if dest_info.get("is_eu") else "red", icon="flag")).add_to(m)
        st_folium(m, width=None, height=400)
    with c_blueprint: st.markdown(render_blueprint_card(st.session_state['p_ship_type'], SHIPS[st.session_state['p_ship_type']], COMMODITY_SPECS[selected_name]), unsafe_allow_html=True)

# --- V3: PROFIT & LOSS ANALYSIS ---
elif st.session_state['current_view'] == "Profit & Loss Analysis":
    st.header("Financial Simulation & Risk")
    def upd_fuel(): val = get_fuel_market_price(); st.session_state['p_fuel_price'] = st.session_state['_p_fuel_price'] = val
    def upd_sofr(): val = float(macro_data.get('sofr_proxy', 5.5)); st.session_state['p_interest'] = st.session_state['_p_interest'] = val
    def upd_carb(): val = get_carbon_price(); st.session_state['p_carbon_price'] = st.session_state['_p_carbon_price'] = val
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Fuel & Emissions"); st.button("Use Market Price", on_click=upd_fuel)
        st.number_input("Fuel Price ($/mt)", step=10.0, key='_p_fuel_price', on_change=save_state, args=('p_fuel_price', '_p_fuel_price'))
        if PORT_COORDINATES[st.session_state.p_destination].get("is_eu"): st.button("Use Market EUA Price", on_click=upd_carb); st.number_input("EUA Price ($/mt)", step=1.0, key='_p_carbon_price', on_change=save_state, args=('p_carbon_price', '_p_carbon_price'))
    with c2:
        st.subheader("Financing"); st.button("Use SOFR (3M)", on_click=upd_sofr)
        st.number_input("Insurance (%)", step=0.01, key='_p_insurance', on_change=save_state, args=('p_insurance', '_p_insurance'))
        st.number_input("Interest Rate (%)", step=0.1, key='_p_interest', on_change=save_state, args=('p_interest', '_p_interest'))
        st.slider("Equity (%)", 0, 100, key='_p_equity', on_change=save_state, args=('p_equity', '_p_equity'))
    with c3:
        st.subheader("Operational Risk")
        st.slider("Demurrage (Days)", 0, 15, key='_p_demurrage_days', on_change=save_state, args=('p_demurrage_days', '_p_demurrage_days'))
        st.number_input("Demurrage ($/day)", step=1000.0, key='_p_demurrage_rate', on_change=save_state, args=('p_demurrage_rate', '_p_demurrage_rate'))
    route_d = get_route_data(st.session_state['p_origin'], st.session_state['p_destination'])
    m = calculate_voyage_metrics(route_d['distance'], st.session_state['p_ship_type'], st.session_state['p_volume'], st.session_state['p_freight_rate'], st.session_state['p_fuel_price'])
    pnl = calculate_final_pnl(st.session_state['p_volume'], st.session_state['p_buy_price'], st.session_state['p_sell_price'], m['freight_total_cost'], m['fuel_cost'], calculate_carbon_cost(st.session_state['p_destination'], m['fuel_burn_total'], st.session_state['p_carbon_price']), st.session_state['p_insurance'], st.session_state['p_interest'], m['days'], st.session_state['p_demurrage_days'], st.session_state['p_demurrage_rate'], equity_pct=st.session_state['p_equity'])
    st.markdown("---"); k1, k2, k3 = st.columns(3); k1.metric("Net Profit", f"${pnl['total_profit']:,.0f}"); k2.metric("ROI", f"{pnl['roi']:.2f}%"); k3.metric("Fuel Burn", f"{m['fuel_burn_total']:.0f} mt")
    fig_w = go.Figure(go.Waterfall(x=["Margin", "Freight", "Fuel", "Carbon", "Insurance", "Finance", "Profit"], y=[pnl['revenue']-pnl['cargo_cost'], -pnl['freight_cost'], -pnl['fuel_cost'], -pnl['carbon_cost'], -pnl['insurance_cost'], -pnl['finance_cost'], pnl['total_profit']]))
    st.plotly_chart(fig_w.update_layout(title="P&L Breakdown Waterfall", template="plotly_dark"), use_container_width=True)
    st.subheader("Sensitivity Analysis (Stress Test)"); fixed = {k: pnl[k] for k in ['cargo_cost', 'fuel_cost', 'carbon_cost', 'finance_cost', 'demurrage_cost']}
    sens = calculate_sensitivity_grid(st.session_state.p_volume, st.session_state.p_sell_price, m['freight_total_cost'], fixed, st.session_state.p_insurance)
    col_heat, col_surf = st.columns(2)
    with col_heat:
        st.markdown("##### Matrix (Heatmap): Sell Price (X) vs Freight (Y)")
        z_text = [[f"${v/1e6:.1f}M" if abs(v) >= 1e6 else f"${v/1e3:.0f}k" for v in row] for row in sens['z_matrix']]
        st.plotly_chart(go.Figure(go.Heatmap(z=sens['z_matrix'], x=sens['x_labels'], y=sens['y_labels'], colorscale='RdYlGn', text=z_text, texttemplate="%{text}")).update_layout(template="plotly_dark", height=400, xaxis_title="Sell Price Delta", yaxis_title="Freight Delta"), use_container_width=True)
    with col_surf:
        st.markdown("##### Risk Topography (3D): Price (X) vs Freight (Y) vs Profit (Z)")
        st.plotly_chart(go.Figure(go.Surface(z=sens['z_matrix'], x=sens['x_labels'], y=sens['y_labels'], colorscale='RdYlGn')).update_layout(template="plotly_dark", height=400), use_container_width=True)
    st.subheader("Monte Carlo Simulation")
    sim = calculate_monte_carlo_var(st.session_state['p_sell_price'], market_data['volatility'] if market_data else 0.3, m['days'], market_data['drift_annual'] if market_data else 0.0)
    sim_profits = (sim["simulated_prices"] * st.session_state['p_volume']) - pnl['total_costs']
    var_95, mean_p = np.percentile(sim_profits, 5), np.mean(sim_profits)
    mc1, mc2 = st.columns([1, 3])
    with mc1: 
        st.metric("VaR (95%)", f"${var_95:,.0f}"); st.metric("Expected Mean", f"${mean_p:,.0f}")
        st.caption(f"Calculated for **{m['days']:.1f}** voyage days using **past monthly asset drift** ({market_data['drift_annual']*100/12:.2f}% monthly avg).")
    with mc2: st.plotly_chart(px.histogram(x=sim_profits, nbins=50).add_vline(x=var_95, line_dash="dash", line_color="red").update_layout(template="plotly_dark", height=300, xaxis_title="Net Profit Distribution"), use_container_width=True)

# --- V4: ARBITRAGE SCANNER ---
elif st.session_state['current_view'] == "Arbitrage Scanner":
    st.header("Global Arbitrage Intelligence")
    st.caption("Scanning combinations of all assets and routes to identify trades with peak ROI.")

    # Confirmation via st.toast (bottom-floating) to ensure visibility while scrolled
    if st.session_state.get('show_arb_ribbon'):
        st.toast(f"Trade Configuration Loaded: {st.session_state.get('last_loaded_trade')}")
        st.session_state['show_arb_ribbon'] = False

    opportunities = []
    fuel_scan_price = get_fuel_market_price()
    carbon_scan_price = get_carbon_price()

    with st.spinner("Analyzing cross-asset route profitability..."):
        for name, ticker in COMMODITIES.items():
            m_data = get_market_data(ticker)
            if not m_data: continue
            buy_p = m_data['price']
            sell_p = buy_p * 1.05
            req_cargo = COMMODITY_SPECS[name]
            for (orig, dest), r_data in ROUTES_DB.items():
                ship_list = [n for n, s in SHIPS.items() if req_cargo in s["category"]]
                if not ship_list: continue
                s_type = ship_list[0]
                vol = SHIPS[s_type]["capacity"]
                m_scan = calculate_voyage_metrics(r_data['dist'], s_type, vol, 0.0, fuel_scan_price)
                pnl_scan = calculate_final_pnl(vol, buy_p, sell_p, m_scan['freight_total_cost'], m_scan['fuel_cost'], calculate_carbon_cost(dest, m_scan['fuel_burn_total'], carbon_scan_price), st.session_state.p_insurance, st.session_state.p_interest, m_scan['days'], 0, 35000, st.session_state.p_equity)
                opportunities.append({
                    "Commodity": name, "Route": f"{orig} → {dest}", "Origin": orig, "Destination": dest,
                    "Ship": s_type, "Vol": vol, "Buy": buy_p, "Sell": sell_p, "Profit": pnl_scan['total_profit'], "ROI": pnl_scan['roi']
                })

    if opportunities:
        df_arb = pd.DataFrame(opportunities).sort_values(by="ROI", ascending=False).head(10)
        for i, row in df_arb.iterrows():
            with st.container():
                c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                c1.markdown(f"**{row['Commodity']}** \n{row['Route']}")
                c2.metric("ROI", f"{row['ROI']:.2f}%")
                c3.metric("Profit", f"${row['Profit']:,.0f}")
                if c4.button("Load Trade", key=f"arb_{i}"):
                    st.session_state['selected_asset'] = row['Commodity']
                    st.session_state['p_origin'] = row['Origin']
                    st.session_state['p_destination'] = row['Destination']
                    st.session_state['p_ship_type'] = row['Ship']
                    st.session_state['p_volume'] = row['Vol']
                    st.session_state['p_buy_price'] = row['Buy']
                    st.session_state['p_sell_price'] = row['Sell']
                    st.session_state['p_fuel_price'] = fuel_scan_price
                    st.session_state['show_arb_ribbon'] = True
                    st.session_state['last_loaded_trade'] = f"{row['Commodity']} ({row['Route']})"
                    st.rerun()
                st.markdown("---")
