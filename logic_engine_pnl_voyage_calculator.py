import numpy as np
import pandas as pd

# Database: Commodities and classification
COMMODITY_SPECS = {
    "Brent Crude": "Oil", "WTI Crude": "Oil", "Dubai Crude": "Oil", "Heating Oil": "Oil",
    "Natural Gas": "Gas", 
    "Corn": "Dry Bulk", "Soybean": "Dry Bulk", "Wheat": "Dry Bulk", "Sugar": "Dry Bulk", "Copper": "Dry Bulk",
    "Gold": "Secure", "Silver": "Secure"
}

# Database: Vessel specifications and dimensions
SHIPS = {
    "VLCC (Crude)": {
        "capacity": 2_000_000, "speed": 13.0, "base_rate": 35000, "fuel_burn": 60.0, 
        "category": ["Oil"], "loa": 330, "beam": 60, "draft": 22.5
    },
    "Suezmax (Crude)": {
        "capacity": 1_000_000, "speed": 13.5, "base_rate": 25000, "fuel_burn": 45.0, 
        "category": ["Oil"], "loa": 274, "beam": 48, "draft": 17.0
    },
    "Aframax (Crude/Prods)": {
        "capacity": 700_000, "speed": 14.0, "base_rate": 20000, "fuel_burn": 35.0, 
        "category": ["Oil"], "loa": 245, "beam": 42, "draft": 15.0
    },
    "MR Tanker (Products)": {
        "capacity": 350_000, "speed": 14.0, "base_rate": 15000, "fuel_burn": 25.0, 
        "category": ["Oil"], "loa": 183, "beam": 32, "draft": 12.0
    },
    "LNG Carrier": {
        "capacity": 3_400_000, "speed": 19.5, "base_rate": 65000, "fuel_burn": 70.0, 
        "category": ["Gas"], "loa": 290, "beam": 46, "draft": 11.5
    },
    "Capesize (Dry Bulk)": {
        "capacity": 1_200_000, "speed": 12.0, "base_rate": 18000, "fuel_burn": 50.0, 
        "category": ["Dry Bulk"], "loa": 292, "beam": 45, "draft": 18.0
    },
    "Panamax (Dry Bulk)": {
        "capacity": 500_000, "speed": 13.0, "base_rate": 14000, "fuel_burn": 30.0, 
        "category": ["Dry Bulk"], "loa": 225, "beam": 32, "draft": 12.0
    },
    "Secure Air/Sea Cargo": {
        "capacity": 5000, "speed": 25.0, "base_rate": 10000, "fuel_burn": 10.0, 
        "category": ["Secure"], "loa": 50, "beam": 10, "draft": 4.0
    }
}

# GIS Database: Maritime chokepoints
CHOKEPOINTS = {
    "Panama Canal": [9.101, -79.695], "Suez Canal": [30.605, 32.324], "Strait of Gibraltar": [35.954, -5.607],
    "Strait of Malacca": [3.194, 100.264], "Strait of Hormuz": [26.566, 56.478], "Cape of Good Hope": [-34.825, 19.015],
    "Bab el-Mandeb": [12.583, 43.333]
}

# GIS Database: Commercial ports and EU ETS status
PORT_COORDINATES = {
    "Houston (US)": {"coords": [29.3, -94.8], "is_eu": False},
    "New York (US)": {"coords": [40.6, -74.0], "is_eu": False},
    "Santos (Brazil)": {"coords": [-23.9, -46.3], "is_eu": False},
    "Vancouver (Canada)": {"coords": [49.3, -123.1], "is_eu": False},
    "Rotterdam (ARA)": {"coords": [51.9, 4.0], "is_eu": True},
    "London (UK)": {"coords": [51.5, 0.5], "is_eu": True},
    "Genoa (Italy)": {"coords": [44.4, 8.9], "is_eu": True},
    "Ras Tanura (Saudi)": {"coords": [26.7, 50.2], "is_eu": False},
    "Fujairah (UAE)": {"coords": [25.1, 56.3], "is_eu": False},
    "Bonny (Nigeria)": {"coords": [4.3, 7.1], "is_eu": False},
    "Richards Bay (SA)": {"coords": [-28.8, 32.0], "is_eu": False},
    "Shanghai (China)": {"coords": [31.2, 122.0], "is_eu": False},
    "Ningbo (China)": {"coords": [29.9, 122.0], "is_eu": False},
    "Singapore": {"coords": [1.2, 103.8], "is_eu": False},
    "Tokyo (Japan)": {"coords": [35.6, 139.8], "is_eu": False}
}

# Navigation waypoints
WAYPOINTS = {
    "Hawaii_Nav": [21.3, -157.8], "Shanghai_West_Virtual": [31.2, -238.0],
    "Indian_Ocean_Mid": [5.0, 80.0], "Atlantic_Mid_South": [-10.0, -10.0], "Cape_Verde_Nav": [16.0, -24.0]
}

# Logistics: Routing database
ROUTES_DB = {
    ("Houston (US)", "Rotterdam (ARA)"): { "dist": 4850, "path": ["Houston (US)", "Rotterdam (ARA)"] },
    ("New York (US)", "Rotterdam (ARA)"): { "dist": 3400, "path": ["New York (US)", "Rotterdam (ARA)"] },
    ("Bonny (Nigeria)", "Rotterdam (ARA)"): { "dist": 4300, "path": ["Bonny (Nigeria)", "Cape_Verde_Nav", "Rotterdam (ARA)"] },
    ("Rotterdam (ARA)", "Shanghai (China)"): { "dist": 10500, "path": ["Rotterdam (ARA)", "Strait of Gibraltar", "Suez Canal", "Bab el-Mandeb", "Indian_Ocean_Mid", "Strait of Malacca", "Shanghai (China)"] },
    ("Rotterdam (ARA)", "Tokyo (Japan)"): { "dist": 11200, "path": ["Rotterdam (ARA)", "Strait of Gibraltar", "Suez Canal", "Bab el-Mandeb", "Indian_Ocean_Mid", "Strait of Malacca", "Tokyo (Japan)"] },
    ("Ras Tanura (Saudi)", "Rotterdam (ARA)"): { "dist": 6300, "path": ["Ras Tanura (Saudi)", "Strait of Hormuz", "Bab el-Mandeb", "Suez Canal", "Strait of Gibraltar", "Rotterdam (ARA)"] },
    ("New York (US)", "Shanghai (China)"): { "dist": 10600, "path": ["New York (US)", "Panama Canal", "Hawaii_Nav", "Shanghai_West_Virtual"] },
    ("Houston (US)", "Shanghai (China)"): { "dist": 9800, "path": ["Houston (US)", "Panama Canal", "Hawaii_Nav", "Shanghai_West_Virtual"] },
    ("Santos (Brazil)", "Shanghai (China)"): { "dist": 11000, "path": ["Santos (Brazil)", "Atlantic_Mid_South", "Cape of Good Hope", "Indian_Ocean_Mid", "Strait of Malacca", "Shanghai (China)"] },
    ("Bonny (Nigeria)", "Shanghai (China)"): { "dist": 9500, "path": ["Bonny (Nigeria)", "Cape of Good Hope", "Indian_Ocean_Mid", "Strait of Malacca", "Shanghai (China)"] },
    ("Ras Tanura (Saudi)", "Shanghai (China)"): { "dist": 5800, "path": ["Ras Tanura (Saudi)", "Strait of Hormuz", "Indian_Ocean_Mid", "Strait of Malacca", "Shanghai (China)"] }
}

def get_route_data(origin, destination):
    """Calculate distance and coordinate path for a route."""
    route = ROUTES_DB.get((origin, destination))
    if not route:
        route = ROUTES_DB.get((destination, origin))
        if route:
            route = route.copy()
            route["path"] = route["path"][::-1]
    origin_coords = PORT_COORDINATES.get(origin, {}).get("coords", [0,0])
    dest_coords = PORT_COORDINATES.get(destination, {}).get("coords", [0,0])
    if not route: return {"distance": 5000, "waypoints": [origin_coords, dest_coords]}
    coords_path = []
    if route["path"][0] != origin: coords_path.append(origin_coords)
    for point_name in route["path"]:
        if point_name in PORT_COORDINATES: coords_path.append(PORT_COORDINATES[point_name]["coords"])
        elif point_name in CHOKEPOINTS: coords_path.append(CHOKEPOINTS[point_name])
        elif point_name in WAYPOINTS: coords_path.append(WAYPOINTS[point_name])
    if "end_coord" in route: coords_path.append(route["end_coord"])
    elif route["path"][-1] != destination: coords_path.append(dest_coords)
    return {"distance": route["dist"], "waypoints": coords_path}

def calculate_voyage_metrics(distance, ship_type, volume, freight_rate_market, fuel_price_per_mt):
    """Calculate days, freight costs and fuel burn."""
    ship = SHIPS.get(ship_type, SHIPS["Aframax (Crude/Prods)"])
    days_steaming = distance / (ship["speed"] * 24)
    total_days = days_steaming + 3
    if freight_rate_market > 0: total_freight_cost = freight_rate_market * volume
    else: total_freight_cost = total_days * ship["base_rate"]
    total_fuel_burn = total_days * ship.get("fuel_burn", 40.0)
    fuel_cost = total_fuel_burn * fuel_price_per_mt
    return {"days": total_days, "freight_total_cost": total_freight_cost, "fuel_cost": fuel_cost, "fuel_burn_total": total_fuel_burn, "base_rate": ship["base_rate"]}

def calculate_carbon_cost(destination, total_fuel_burn, carbon_price):
    """Calculate carbon liability for EU ETS zones."""
    port_info = PORT_COORDINATES.get(destination, {})
    if port_info.get("is_eu", False): return total_fuel_burn * 3.114 * carbon_price
    return 0.0

def calculate_final_pnl(volume, price_buy, price_sell, freight_cost, fuel_cost, carbon_cost, insurance_rate, finance_rate, voyage_days, demurrage_days, demurrage_rate, equity_pct=100):
    """Waterfall P&L calculation including financing and insurance."""
    revenue = volume * price_sell
    cargo_cost = volume * price_buy
    debt_share = 1 - (equity_pct / 100)
    financed_amount = cargo_cost * debt_share
    finance_cost = (financed_amount * (finance_rate / 100) * voyage_days) / 360
    insurance_cost = revenue * (insurance_rate / 100)
    demurrage_cost = demurrage_days * demurrage_rate
    total_costs = cargo_cost + freight_cost + fuel_cost + carbon_cost + insurance_cost + finance_cost + demurrage_cost
    profit = revenue - total_costs
    return {
        "revenue": revenue, "cargo_cost": cargo_cost, 
        "freight_cost": freight_cost, "fuel_cost": fuel_cost, "carbon_cost": carbon_cost,
        "insurance_cost": insurance_cost, "finance_cost": finance_cost, "demurrage_cost": demurrage_cost,
        "total_profit": profit, "roi": (profit / total_costs) * 100 if total_costs > 0 else 0, "total_costs": total_costs
    }

def calculate_monte_carlo_var(current_price, volatility_annual, days, annual_drift_pct=0.0, iterations=1000):
    """Probabilistic Value at Risk calculation."""
    if volatility_annual is None or volatility_annual == 0: volatility_annual = 0.30 
    volatility_voyage = volatility_annual * np.sqrt(days / 365.0)
    drift_voyage = annual_drift_pct * (days / 365.0)
    simulated_returns = np.random.normal(drift_voyage, volatility_voyage, iterations)
    simulated_prices = current_price * (1 + simulated_returns)
    var_95_price = np.percentile(simulated_prices, 5)
    return { "simulated_prices": simulated_prices, "var_95_price": var_95_price, "mean_price": np.mean(simulated_prices) }

def calculate_sensitivity_grid(volume, base_sell_price, base_freight_cost, fixed_costs_dict, insurance_rate, price_range_pct=0.10, freight_range_pct=0.20, steps=9):
    """Generate profit matrix for heatmap and surface analysis."""
    price_multipliers = np.linspace(1 - price_range_pct, 1 + price_range_pct, steps)
    freight_multipliers = np.linspace(1 - freight_range_pct, 1 + freight_range_pct, steps)
    price_labels = [f"{(m-1)*100:+.0f}%" for m in price_multipliers]
    freight_labels = [f"{(m-1)*100:+.0f}%" for m in freight_multipliers]
    z_matrix = []
    total_fixed = fixed_costs_dict['cargo_cost'] + fixed_costs_dict['fuel_cost'] + fixed_costs_dict['carbon_cost'] + fixed_costs_dict['finance_cost'] + fixed_costs_dict['demurrage_cost']
    for f_mult in freight_multipliers:
        row = []
        sim_freight_cost = base_freight_cost * f_mult
        for p_mult in price_multipliers:
            sim_sell_price = base_sell_price * p_mult
            sim_revenue = volume * sim_sell_price
            sim_insurance = sim_revenue * (insurance_rate / 100)
            sim_profit = sim_revenue - (total_fixed + sim_freight_cost + sim_insurance)
            row.append(sim_profit)
        z_matrix.append(row)
    return {
        "x_labels": price_labels, "y_labels": freight_labels, "z_matrix": z_matrix,
        "x_values": [base_sell_price * m for m in price_multipliers], "y_values": [base_freight_cost * m for m in freight_multipliers]
    }