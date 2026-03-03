import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium

from data_engine_pnl_voyage_calculator import (
    COMMODITIES, get_market_data, get_correlations,
    get_macro_data, get_carbon_price, get_fuel_market_price,
    get_forward_curve, get_front_month_price, get_seasonality_data,
    get_matched_forward_price
)
from logic_engine_pnl_voyage_calculator import (
    PORT_COORDINATES, CHOKEPOINTS, ROUTES_DB, SHIPS, COMMODITY_SPECS,
    COMMODITY_UNITS,
    get_route_data, calculate_voyage_metrics, calculate_carbon_cost,
    calculate_canal_fees, calculate_final_pnl,
    calculate_monte_carlo_var, calculate_sensitivity_grid,
    detect_curve_structure, calculate_calendar_arb
)

# --- Configuration and CSS ---
st.set_page_config(page_title="P&L Voyage Calculator", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stMarkdown, .stHeader, .stSubheader, h1, h2, h3, h4, p, span, div { color: #FAFAFA !important; }
    .stTextInput input, .stNumberInput input { background-color: #2d323d; color: #FAFAFA; border-color: #555; }
    .stSelectbox div[data-baseweb="select"] { background-color: #161920 !important; color: #FAFAFA; border-color: #444; }
    div[data-baseweb="popover"] > div { background-color: #1d2129 !important; border: 1px solid #333 !important; }
    div[role="listbox"] { background-color: #1d2129 !important; }
    div[role="option"] { background-color: #1d2129 !important; color: #FAFAFA !important; }
    div[role="option"]:hover, div[role="option"][aria-selected="true"] { background-color: #1d2129 !important; border: none !important; outline: none !important; }
    div[data-testid="stToast"] { background-color: #161920 !important; color: #FAFAFA !important; border: 1px solid #333; }
    div[data-testid="stMetric"] { background-color: #161920; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    div[data-testid="stMetricLabel"] { color: #aaa !important; }
    div[data-testid="stMetricValue"] { color: #FAFAFA !important; }
    .leaflet-control-attribution { display: none !important; }
    html, body, [class*="css"] { font-size: 16px; }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
def init_state(key, default_value):
    if key not in st.session_state:
        st.session_state[key] = default_value

def save_state(perm_key, widget_key):
    st.session_state[perm_key] = st.session_state[widget_key]

init_state('current_view', 'Market Intelligence')
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
init_state('p_hedge_ratio', 0)
init_state('p_hedge_price', 0.0)

for k in ['p_origin', 'p_destination', 'p_ship_type', 'p_volume',
          'p_buy_price', 'p_sell_price', 'p_freight_rate', 'p_fuel_price',
          'p_carbon_price', 'p_interest', 'p_insurance', 'p_equity',
          'p_demurrage_days', 'p_demurrage_rate', 'p_hedge_ratio', 'p_hedge_price']:
    st.session_state[f'_{k}'] = st.session_state[k]

# --- Navigation (radio for active highlight) ---
st.sidebar.title("P&L Voyage Calculator")
NAV_VIEWS = ["Market Intelligence", "Trade Configuration", "P&L Results", "Contango Carry Scanner"]
nav_idx = NAV_VIEWS.index(st.session_state['current_view']) if st.session_state['current_view'] in NAV_VIEWS else 0
selected_nav = st.sidebar.radio("Navigation", NAV_VIEWS, index=nav_idx, label_visibility="collapsed")
if selected_nav != st.session_state['current_view']:
    st.session_state['current_view'] = selected_nav
    st.rerun()

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
else:
    market_data, macro_data, current_price = None, None, 0.0

# --- UI Helpers ---
def get_unit_label():
    if selected_name and selected_name in COMMODITY_UNITS:
        return COMMODITY_UNITS[selected_name]
    return "units"

def style_correlation_table(df):
    return df.style.background_gradient(cmap='RdYlGn', subset=['Correlation'], vmin=-1, vmax=1).format({"Correlation": "{:.2f}"})

def render_blueprint_card(ship_type, s_data, cargo_type):
    n_loa = min(s_data['loa'] / 350 * 100, 100)
    n_speed = min(s_data['speed'] / 25 * 100, 100)
    n_fuel = min(s_data['fuel_burn'] / 80 * 100, 100)
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
        <div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; font-size: 12px; color: #888;"><span>SIZE</span><span>{s_data.get('loa')}m</span></div><div style="height: 6px; background: #333; border-radius: 3px; overflow: hidden;"><div style="width: {n_loa}%; height: 100%; background: #00CC96;"></div></div></div>
        <div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; font-size: 12px; color: #888;"><span>SPEED</span><span>{s_data.get('speed')} kts</span></div><div style="height: 6px; background: #333; border-radius: 3px; overflow: hidden;"><div style="width: {n_speed}%; height: 100%; background: #3388ff;"></div></div></div>
        <div style="margin-bottom: 10px;"><div style="display: flex; justify-content: space-between; font-size: 12px; color: #888;"><span>FUEL</span><span>{s_data.get('fuel_burn')} mt/d</span></div><div style="height: 6px; background: #333; border-radius: 3px; overflow: hidden;"><div style="width: {n_fuel}%; height: 100%; background: #EF553B;"></div></div></div>
    </div>
</div>"""

# ===================================================================
# V1: MARKET INTELLIGENCE
# ===================================================================
if st.session_state['current_view'] == "Market Intelligence":
    st.subheader("Asset Selection")
    try:
        current_idx = list(COMMODITIES.keys()).index(st.session_state['selected_asset'])
    except:
        current_idx = None
    def update_asset():
        st.session_state['p_buy_price'] = st.session_state['p_sell_price'] = 0.0
        st.session_state['p_hedge_price'] = 0.0
        st.session_state['selected_asset'] = st.session_state['_selected_asset']
    st.selectbox("Select Commodity", list(COMMODITIES.keys()), index=current_idx, key='_selected_asset', on_change=update_asset, placeholder="Choose an asset...", label_visibility="collapsed")
    if not st.session_state['selected_asset']:
        st.info("Please select a commodity from the dropdown above."); st.stop()

    if market_data:
        unit = get_unit_label()
        c1, c2, c3 = st.columns(3)
        c1.metric("Last Price", f"${market_data['price']:,.2f}/{unit}", f"{market_data['delta_1d']*100:.2f}%")
        c2.metric("Weekly Change", f"{market_data['delta_1w']*100:.2f}%")
        c3.metric("Monthly Change", f"{market_data['delta_1m']*100:.2f}%")

        st.subheader("Price History")
        fig = px.line(market_data['history'], title=f"{selected_name} Trend")
        fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # --- Forward Curve ---
        st.subheader("Forward Curve")
        fc1, fc2 = st.columns([2, 1])
        with st.spinner("Fetching futures strip..."):
            fwd_curve = get_forward_curve(selected_name)
        with fc1:
            if fwd_curve:
                fwd_df = pd.DataFrame(fwd_curve)
                curve_info = detect_curve_structure(fwd_curve)
                color = "#00CC96" if curve_info["carry_pct"] >= 0 else "#EF553B"
                fig_curve = px.line(fwd_df, x="label", y="price", title=f"{selected_name} Futures Strip (Live)", markers=True)
                fig_curve.update_traces(line_color=color, line_width=3)
                fig_curve.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', xaxis_title="Contract Month", yaxis_title=f"Price ($/{unit})")
                st.plotly_chart(fig_curve, use_container_width=True)
                st.caption("Source: Yahoo Finance — exchange-traded futures (CME/NYMEX/CBOT/COMEX).")
            else:
                months = np.arange(1, 7)
                proj = [current_price * np.exp(market_data['drift_annual'] * (m / 12)) for m in months]
                fig_curve = px.line(pd.DataFrame({"Month": [f"M+{m}" for m in months], "Price": proj}), x="Month", y="Price", title="Estimated Forward Curve (Momentum-Based)", markers=True)
                color = "#00CC96" if market_data['drift_annual'] >= 0 else "#EF553B"
                fig_curve.update_traces(line_color=color, line_width=3)
                fig_curve.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_curve, use_container_width=True)
                st.caption("Futures strip unavailable — showing momentum-based estimate.")
        with fc2:
            st.markdown("##### Structure")
            if fwd_curve:
                curve_info = detect_curve_structure(fwd_curve)
                if curve_info["structure"] == "Contango":
                    st.info(f"Contango ({curve_info['carry_pct']:+.1f}% ann.)")
                elif curve_info["structure"] == "Backwardation":
                    st.warning(f"Backwardation ({curve_info['carry_pct']:+.1f}% ann.)")
                else:
                    st.caption("Flat curve")
                st.caption(f"Front: ${curve_info['front']:,.2f} / Back: ${curve_info['back']:,.2f}")
            else:
                drift_6m = (market_data['drift_annual'] / 2) * 100
                if market_data['drift_annual'] > 0.02:
                    st.info(f"Momentum suggests contango (+{drift_6m:.1f}%)")
                elif market_data['drift_annual'] < -0.02:
                    st.warning(f"Momentum suggests backwardation ({drift_6m:.1f}%)")
                else:
                    st.caption("Neutral")

        # --- Seasonality ---
        st.subheader("Seasonality Analysis")
        with st.spinner("Loading 10Y seasonality data..."):
            season_data = get_seasonality_data(selected_ticker)
        if season_data is not None:
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            fig_season = go.Figure()
            fig_season.add_trace(go.Scatter(x=month_names, y=season_data["max"], mode='lines', name="10Y Max", line=dict(width=0), showlegend=False))
            fig_season.add_trace(go.Scatter(x=month_names, y=season_data["min"], mode='lines', name="10Y Range", fill='tonexty', fillcolor='rgba(51,136,255,0.15)', line=dict(width=0)))
            fig_season.add_trace(go.Scatter(x=month_names, y=season_data["avg"], mode='lines+markers', name="10Y Average", line=dict(color="#3388ff", width=3)))
            if season_data["current_year"] is not None:
                cy = season_data["current_year"]
                cy_months = [month_names[m - 1] for m in cy.index]
                fig_season.add_trace(go.Scatter(x=cy_months, y=cy.values, mode='lines+markers', name=str(season_data["current_year_label"]), line=dict(color="#EF553B", width=3, dash="dot")))
            fig_season.update_layout(title=f"{selected_name} — Monthly Avg Price (10Y vs Current Year)", template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title=f"Price ($/{unit})", xaxis_title="Month")
            st.plotly_chart(fig_season, use_container_width=True)
        else:
            st.caption("Seasonality data unavailable.")

        # --- Correlations ---
        st.subheader("Asset Correlations")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.write("##### 1 Week")
            c1w = get_correlations(selected_ticker, period="1wk")
            if not c1w.empty: st.dataframe(style_correlation_table(c1w.head(5)), use_container_width=True)
        with col_c2:
            st.write("##### 1 Month")
            c1m = get_correlations(selected_ticker, period="1mo")
            if not c1m.empty: st.dataframe(style_correlation_table(c1m.head(5)), use_container_width=True)
        with col_c3:
            st.write("##### 1 Year")
            c1y = get_correlations(selected_ticker, period="1y")
            if not c1y.empty: st.dataframe(style_correlation_table(c1y.head(5)), use_container_width=True)
        with col_c4:
            st.write("##### 10 Year")
            c10y = get_correlations(selected_ticker, period="10y")
            if not c10y.empty: st.dataframe(style_correlation_table(c10y.head(5)), use_container_width=True)

# ===================================================================
# V2: TRADE CONFIGURATION (4 separate rows)
# ===================================================================
elif st.session_state['current_view'] == "Trade Configuration":
    if not st.session_state['selected_asset']:
        st.info("Please select a commodity in Market Intelligence first."); st.stop()

    unit = get_unit_label()
    ports = sorted(list(set([k[0] for k in ROUTES_DB.keys()] + [k[1] for k in ROUTES_DB.keys()])))

    # --- Row 1: Map + Blueprint + Logistics ---
    st.subheader("1. Logistics")
    c_map, c_blueprint, c_logistics = st.columns([2, 1, 1])

    with c_logistics:
        st.selectbox("Loading Port", ports, index=ports.index(st.session_state.p_origin), key='_p_origin', on_change=save_state, args=('p_origin', '_p_origin'))
        st.selectbox("Discharge Port", ports, index=ports.index(st.session_state.p_destination), key='_p_destination', on_change=save_state, args=('p_destination', '_p_destination'))
        comp_ships = [n for n, s in SHIPS.items() if COMMODITY_SPECS[selected_name] in s["category"]]
        if st.session_state['p_ship_type'] not in comp_ships: st.session_state['p_ship_type'] = comp_ships[0]
        st.selectbox("Vessel Class", comp_ships, index=comp_ships.index(st.session_state.p_ship_type), key='_p_ship_type', on_change=save_state, args=('p_ship_type', '_p_ship_type'))
        st.number_input(f"Volume ({unit})", step=1000, key='_p_volume', on_change=save_state, args=('p_volume', '_p_volume'))

    route_data = get_route_data(st.session_state['p_origin'], st.session_state['p_destination'])
    canal_info = calculate_canal_fees(route_data.get("path_names", []), st.session_state['p_ship_type'])

    with c_map:
        wps = route_data['waypoints']; center = wps[len(wps) // 2] if wps else [20, 0]
        m = folium.Map(location=center, zoom_start=2, tiles="CartoDB dark_matter", attr=' ', attribution_control=False)
        if wps:
            folium.PolyLine(locations=wps, color="#3388ff", weight=3, opacity=0.8, dash_array='5, 10').add_to(m)
            folium.Marker(wps[0], popup="START", icon=folium.Icon(color="green", icon="play")).add_to(m)
            dest_info = PORT_COORDINATES.get(st.session_state['p_destination'], {})
            if dest_info.get("coords"):
                folium.Marker(dest_info["coords"], popup="END", icon=folium.Icon(color="blue" if dest_info.get("is_eu") else "red", icon="flag")).add_to(m)
        st_folium(m, width=None, height=350)
        if canal_info["total"] > 0:
            for canal, fee in canal_info["detail"].items():
                st.caption(f"Canal fee — {canal}: **${fee:,.0f}**")

    with c_blueprint:
        st.markdown(render_blueprint_card(st.session_state['p_ship_type'], SHIPS[st.session_state['p_ship_type']], COMMODITY_SPECS[selected_name]), unsafe_allow_html=True)

    # --- Row 2: Trade Pricing ---
    st.markdown("---")
    st.subheader("2. Trade Pricing")
    tp1, tp2, tp3 = st.columns(3)
    with tp1:
        st.number_input(f"FOB Buy ($/{unit})", step=0.01, key='_p_buy_price', on_change=save_state, args=('p_buy_price', '_p_buy_price'))
    with tp2:
        st.number_input(f"CIF Sell ($/{unit})", step=0.01, key='_p_sell_price', on_change=save_state, args=('p_sell_price', '_p_sell_price'))
    with tp3:
        st.number_input("Charter Rate ($ lump sum)", step=1000.0, key='_p_freight_rate', on_change=save_state, args=('p_freight_rate', '_p_freight_rate'))
        st.caption("Set to 0 for vessel class auto-rate (TCE).")

    # --- Row 3: Hedging ---
    st.markdown("---")
    st.subheader("3. Hedging")
    hd1, hd2, hd3 = st.columns(3)
    with hd1:
        st.slider("Hedge Ratio (%)", 0, 100, key='_p_hedge_ratio', on_change=save_state, args=('p_hedge_ratio', '_p_hedge_ratio'))
    with hd2:
        st.number_input(f"Hedge Price ($/{unit})", step=0.01, key='_p_hedge_price', on_change=save_state, args=('p_hedge_price', '_p_hedge_price'))
    with hd3:
        def _load_front_month():
            fm = get_front_month_price(selected_name)
            if fm: st.session_state['p_hedge_price'] = float(fm)
        st.button("Use Front-Month Futures", on_click=_load_front_month)
        st.caption("Hedged volume sells at locked futures price. Unhedged exposed to CIF sell at discharge.")

    # --- Row 4: Costs & Financing ---
    st.markdown("---")
    st.subheader("4. Costs & Financing")
    def upd_fuel(): st.session_state['p_fuel_price'] = get_fuel_market_price()
    def upd_sofr(): st.session_state['p_interest'] = float(macro_data.get('sofr_proxy', 5.5))
    def upd_carb(): st.session_state['p_carbon_price'] = get_carbon_price()

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown("##### Fuel & Emissions")
        st.button("Use Market Fuel Price", on_click=upd_fuel)
        st.number_input("Fuel Price ($/mt)", step=10.0, key='_p_fuel_price', on_change=save_state, args=('p_fuel_price', '_p_fuel_price'))
        origin_eu = PORT_COORDINATES[st.session_state.p_origin].get("is_eu", False)
        dest_eu = PORT_COORDINATES[st.session_state.p_destination].get("is_eu", False)
        if origin_eu or dest_eu:
            st.button("Use Market EUA Price", on_click=upd_carb)
            st.number_input("EUA Price ($/tCO2)", step=1.0, key='_p_carbon_price', on_change=save_state, args=('p_carbon_price', '_p_carbon_price'))
            ets_pct = "100%" if (origin_eu and dest_eu) else "50%"
            st.caption(f"EU ETS at **{ets_pct}** for this route.")
    with cc2:
        st.markdown("##### Financing")
        st.button("Use SOFR (3M T-Bill)", on_click=upd_sofr)
        st.number_input("Insurance (%)", step=0.01, key='_p_insurance', on_change=save_state, args=('p_insurance', '_p_insurance'))
        st.number_input("Interest Rate (%)", step=0.1, key='_p_interest', on_change=save_state, args=('p_interest', '_p_interest'))
        st.slider("Equity (%)", 0, 100, key='_p_equity', on_change=save_state, args=('p_equity', '_p_equity'))
    with cc3:
        st.markdown("##### Operational Risk")
        st.slider("Demurrage (Days)", 0, 15, key='_p_demurrage_days', on_change=save_state, args=('p_demurrage_days', '_p_demurrage_days'))
        st.number_input("Demurrage ($/day)", step=1000.0, key='_p_demurrage_rate', on_change=save_state, args=('p_demurrage_rate', '_p_demurrage_rate'))

# ===================================================================
# V3: P&L RESULTS
# ===================================================================
elif st.session_state['current_view'] == "P&L Results":
    if not st.session_state['selected_asset']:
        st.info("Please select a commodity in Market Intelligence first."); st.stop()
    st.header("Financial Simulation & Risk")
    unit = get_unit_label()

    route_d = get_route_data(st.session_state['p_origin'], st.session_state['p_destination'])
    canal_info = calculate_canal_fees(route_d.get("path_names", []), st.session_state['p_ship_type'])
    metrics = calculate_voyage_metrics(route_d['distance'], st.session_state['p_ship_type'], st.session_state['p_volume'], st.session_state['p_freight_rate'], st.session_state['p_fuel_price'])
    carbon = calculate_carbon_cost(st.session_state['p_origin'], st.session_state['p_destination'], metrics['fuel_burn_total'], st.session_state['p_carbon_price'])
    pnl = calculate_final_pnl(st.session_state['p_volume'], st.session_state['p_buy_price'], st.session_state['p_sell_price'], metrics['freight_total_cost'], metrics['fuel_cost'], carbon, st.session_state['p_insurance'], st.session_state['p_interest'], metrics['days'], st.session_state['p_demurrage_days'], st.session_state['p_demurrage_rate'], canal_fees_total=canal_info["total"], equity_pct=st.session_state['p_equity'], hedge_ratio=st.session_state['p_hedge_ratio'], hedge_price=st.session_state['p_hedge_price'])

    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Net Profit", f"${pnl['total_profit']:,.0f}"); k2.metric("ROI", f"{pnl['roi']:.2f}%"); k3.metric("Voyage Days", f"{metrics['days']:.1f}"); k4.metric("Fuel Burn", f"{metrics['fuel_burn_total']:.0f} mt")

    if pnl['hedge_ratio'] > 0:
        st.markdown("---")
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("Hedge Ratio", f"{pnl['hedge_ratio']}%"); hc2.metric("Locked Margin", f"${pnl['locked_margin']:,.0f}"); hc3.metric("Unhedged Exposure", f"${pnl['unhedged_margin']:,.0f}")

    waterfall_x = ["Gross Margin", "Freight", "Fuel", "Canal Fees", "Carbon (ETS)", "Insurance", "Finance", "Demurrage", "Net Profit"]
    waterfall_y = [pnl['revenue'] - pnl['cargo_cost'], -pnl['freight_cost'], -pnl['fuel_cost'], -pnl['canal_fees'], -pnl['carbon_cost'], -pnl['insurance_cost'], -pnl['finance_cost'], -pnl['demurrage_cost'], pnl['total_profit']]
    fig_w = go.Figure(go.Waterfall(x=waterfall_x, y=waterfall_y, connector={"line": {"color": "#555"}}))
    fig_w.update_layout(title="P&L Breakdown Waterfall", template="plotly_dark", height=450)
    st.plotly_chart(fig_w, use_container_width=True)

    st.subheader("Sensitivity Analysis")
    fixed = {k: pnl[k] for k in ['cargo_cost', 'fuel_cost', 'carbon_cost', 'finance_cost', 'demurrage_cost']}
    fixed['canal_fees'] = pnl['canal_fees']
    sens = calculate_sensitivity_grid(st.session_state.p_volume, st.session_state.p_sell_price, metrics['freight_total_cost'], fixed, st.session_state.p_insurance)
    col_heat, col_surf = st.columns(2)
    with col_heat:
        st.markdown("##### Heatmap: Sell Price vs Freight")
        z_text = [[f"${v/1e6:.1f}M" if abs(v) >= 1e6 else f"${v/1e3:.0f}k" for v in row] for row in sens['z_matrix']]
        fig_h = go.Figure(go.Heatmap(z=sens['z_matrix'], x=sens['x_labels'], y=sens['y_labels'], colorscale='RdYlGn', text=z_text, texttemplate="%{text}"))
        fig_h.update_layout(template="plotly_dark", height=400, xaxis_title="Sell Price Delta", yaxis_title="Freight Delta")
        st.plotly_chart(fig_h, use_container_width=True)
    with col_surf:
        st.markdown("##### 3D Risk Surface")
        fig_s = go.Figure(go.Surface(z=sens['z_matrix'], x=sens['x_labels'], y=sens['y_labels'], colorscale='RdYlGn'))
        fig_s.update_layout(template="plotly_dark", height=400, scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)))
        st.plotly_chart(fig_s, use_container_width=True)

    # --- Monte Carlo VaR & CVaR ---
    st.subheader("Monte Carlo VaR & CVaR")
    vol_ann = market_data['volatility'] if market_data else 0.3
    drift_ann = market_data['drift_annual'] if market_data else 0.0
    kurt = market_data.get('kurtosis', 0.0) if market_data else 0.0
    unhedged_frac = 1.0 - (st.session_state['p_hedge_ratio'] / 100.0)
    effective_vol = st.session_state['p_volume'] * unhedged_frac
    sim = calculate_monte_carlo_var(st.session_state['p_sell_price'], vol_ann, metrics['days'], drift_ann, kurt)
    sim_profits = (sim["simulated_prices"] * effective_vol) - (pnl['total_costs'] * unhedged_frac)
    if st.session_state['p_hedge_ratio'] > 0: sim_profits += pnl['locked_margin']
    var_95 = np.percentile(sim_profits, 5)
    cvar_95 = np.mean(sim_profits[sim_profits <= var_95])
    mean_p = np.mean(sim_profits)

    mc1, mc2 = st.columns([1, 3])
    with mc1:
        st.metric("VaR (95%)", f"${var_95:,.0f}")
        st.metric("CVaR (95%)", f"${cvar_95:,.0f}")
        st.metric("Expected Mean", f"${mean_p:,.0f}")
        st.caption(f"Distribution: **{sim['distribution']}** (df={sim['df_used']:.0f})")
        if st.session_state['p_hedge_ratio'] > 0:
            st.caption(f"Risk on **{unhedged_frac*100:.0f}%** unhedged volume. Locked margin ${pnl['locked_margin']:,.0f} added back.")
    with mc2:
        fig_mc = px.histogram(x=sim_profits, nbins=60, title="Net Profit Distribution")
        fig_mc.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text="VaR 95%", annotation_position="top left", annotation_font_color="red")
        fig_mc.add_vline(x=cvar_95, line_dash="dot", line_color="orange", annotation_text="CVaR 95%", annotation_position="bottom left", annotation_font_color="orange")
        fig_mc.update_layout(template="plotly_dark", height=350, xaxis_title="Net Profit ($)")
        st.plotly_chart(fig_mc, use_container_width=True)

# ===================================================================
# V4: CONTANGO CARRY SCANNER
# ===================================================================
elif st.session_state['current_view'] == "Contango Carry Scanner":
    st.header("Contango Carry Scanner")
    st.caption("Buy physical at spot, ship over N days, sell at the forward price matching the arrival date. "
               "Profit = (forward - spot) x volume - total voyage cost. Fully hedged at inception.")

    if st.session_state.get('show_arb_ribbon'):
        st.toast(f"Trade Loaded: {st.session_state.get('last_loaded_trade')}"); st.session_state['show_arb_ribbon'] = False

    fuel_scan_price = get_fuel_market_price()
    carbon_scan_price = get_carbon_price()
    cal_opps = []
    commodity_status = {}

    with st.spinner("Scanning forward curves across all assets and routes..."):
        for name, ticker in COMMODITIES.items():
            fwd = get_forward_curve(name)
            if not fwd:
                commodity_status[name] = {"status": "no_curve"}
                continue

            m_data = get_market_data(ticker)
            if not m_data:
                commodity_status[name] = {"status": "no_data"}
                continue

            spot_price = float(m_data["price"])
            curve_info = detect_curve_structure(fwd)
            commodity_status[name] = {
                "status": "scanned",
                "structure": curve_info["structure"],
                "carry_pct": curve_info["carry_pct"],
                "spot": spot_price,
                "front": fwd[0]["price"],
                "back": fwd[-1]["price"],
                "has_opportunities": False
            }

            req_cargo = COMMODITY_SPECS[name]
            for (orig, dest), r_data in ROUTES_DB.items():
                ship_list = sorted(
                    [n for n, s in SHIPS.items() if req_cargo in s["category"]],
                    key=lambda n: SHIPS[n]["capacity"])
                if not ship_list: continue
                s_type = ship_list[0]
                vol = SHIPS[s_type]["capacity"]

                route_info = get_route_data(orig, dest)
                canal = calculate_canal_fees(route_info.get("path_names", []), s_type)
                m_scan = calculate_voyage_metrics(r_data['dist'], s_type, vol, 0.0, fuel_scan_price)
                carb = calculate_carbon_cost(orig, dest, m_scan['fuel_burn_total'], carbon_scan_price)
                total_cost = m_scan['freight_total_cost'] + m_scan['fuel_cost'] + carb + canal["total"]

                fwd_price, fwd_label = get_matched_forward_price(name, m_scan['days'])
                if not fwd_price or fwd_price <= spot_price: continue

                arb = calculate_calendar_arb(spot_price, fwd_price, total_cost, vol)
                if arb["margin"] > 0:
                    commodity_status[name]["has_opportunities"] = True
                    cal_opps.append({
                        "Commodity": name, "Route": f"{orig} > {dest}",
                        "Origin": orig, "Destination": dest, "Ship": s_type, "Vol": vol,
                        "Spot": spot_price, "Forward": fwd_price,
                        "Fwd Contract": fwd_label,
                        "Days": f"{m_scan['days']:.0f}d",
                        "Carry": f"{((fwd_price/spot_price)-1)*100:+.2f}%",
                        "Margin": arb["margin"], "ROI": arb["roi"]
                    })

    def _render_opp(row, key_prefix, idx):
        with st.container():
            a1, a2, a3, a4, a5, a6 = st.columns([2, 1, 1, 1, 1, 1])
            a1.markdown(f"**{row['Commodity']}** — {row['Route']}")
            a2.metric("Voyage", row['Days'])
            a3.metric("Carry", row['Carry'])
            a4.metric("Margin", f"${row['Margin']:,.0f}")
            a5.metric("ROI", f"{row['ROI']:.2f}%")
            if a6.button("Load", key=f"{key_prefix}_{idx}"):
                st.session_state['selected_asset'] = row['Commodity']
                st.session_state['p_origin'] = row['Origin']; st.session_state['p_destination'] = row['Destination']
                st.session_state['p_ship_type'] = row['Ship']; st.session_state['p_volume'] = row['Vol']
                st.session_state['p_buy_price'] = row['Spot']; st.session_state['p_sell_price'] = row['Forward']
                st.session_state['p_hedge_ratio'] = 100; st.session_state['p_hedge_price'] = row['Forward']
                st.session_state['p_fuel_price'] = fuel_scan_price
                st.session_state['show_arb_ribbon'] = True
                st.session_state['last_loaded_trade'] = f"{row['Commodity']} Carry ({row['Route']})"; st.rerun()
            st.caption(f"Buy spot ${row['Spot']:,.2f} — Sell {row['Fwd Contract']} fwd ${row['Forward']:,.2f} — Voyage cost deducted")
            st.markdown("---")

    if cal_opps:
        df_all = pd.DataFrame(cal_opps).sort_values(by="ROI", ascending=False)

        # --- Top 3 Overall ---
        st.subheader("Top 3 Carry Trades")
        top3 = df_all.head(3)
        for i, row in top3.iterrows():
            _render_opp(row, "top", i)

        # --- Best by Commodity ---
        st.subheader("Best Opportunity by Commodity")
        best_per = df_all.groupby("Commodity").first().reset_index()
        shown = set(best_per['Commodity'].unique())
        for i, row in best_per.iterrows():
            _render_opp(row, "best", i)

        # Show remaining commodities with no profitable trades
        for name in sorted(set(COMMODITIES.keys()) - shown):
            info = commodity_status.get(name, {})
            if info.get("status") == "no_curve":
                st.caption(f"**{name}** — No futures strip available")
            elif info.get("status") == "scanned":
                st.caption(f"**{name}** — {info['structure']} ({info['carry_pct']:+.1f}% ann.) — Near-month carry insufficient after voyage costs")
            else:
                st.caption(f"**{name}** — Data unavailable")

    else:
        # --- Informative empty state ---
        st.markdown("### No Profitable Carry Trades Found")
        st.markdown("The scanner compared **spot prices** against **voyage-duration-matched forward contracts** "
                    "for all commodities and routes. No combination produced a positive margin after deducting "
                    "freight, fuel, canal fees, carbon, and financing costs.")
        st.markdown("This is normal — physical contango carry opportunities are rare and fleeting. "
                    "They typically emerge during periods of steep contango driven by oversupply or demand shocks.")
        st.markdown("---")
        st.subheader("Market Structure Overview")
        for name in sorted(COMMODITIES.keys()):
            info = commodity_status.get(name, {})
            if info.get("status") == "no_curve":
                st.caption(f"**{name}** — No futures strip available")
            elif info.get("status") == "scanned":
                structure = info["structure"]
                carry = info["carry_pct"]
                spot = info["spot"]
                front = info["front"]
                back = info["back"]
                if structure == "Contango":
                    reason = "Near-month forward at arrival date is below spot, or carry insufficient to cover voyage costs"
                elif structure == "Backwardation":
                    reason = "Curve slopes downward — no carry premium to capture"
                else:
                    reason = "Curve is flat — insufficient spread"
                st.markdown(f"**{name}** — {structure} ({carry:+.1f}% ann.) | Spot: ${spot:,.2f} | Front: ${front:,.2f} | Back: ${back:,.2f}")
                st.caption(f"  {reason}")
            else:
                st.caption(f"**{name}** — Data unavailable")
