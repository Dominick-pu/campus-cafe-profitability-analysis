# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:50:40 2026

@author: jty88
"""


"""Campus Café Profitability Analysis — synthetic POS demonstration (v2).

Run this file directly in Spyder or from a terminal. When the three input CSVs
are ALL absent, it generates reproducible synthetic data. To replace existing
synthetic files, execute with --generate-demo (or first move the data folder).

No real transaction data are included or claimed. All scenario returns are
MODEL-BASED GROSS PROFIT (revenue minus direct item costs), not net profit.
"""

from itertools import combinations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SEED = 2025
FEATURES = ["log_price", "weekend", "promotion_flag", "day_index"]
INPUT_FILES = ["sales_transactions.csv", "products.csv", "inventory_summary.csv"]


def generate_synthetic_data(data_dir, days=180):
    """Produce example orders plus simulated catalog and stock-flow accounting.

    Synthetic basket memberships are assigned randomly; observed demo item
    pairings are NOT evidence of genuine consumer complementarity.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    specs = [
        ("LAT", "Latte", "Beverages", 4.50, 1.45, -1.75, 58),
        ("AME", "Americano", "Beverages", 3.20, 0.80, -1.15, 41),
        ("CAP", "Cappuccino", "Beverages", 4.25, 1.30, -1.55, 43),
        ("MOC", "Mocha", "Beverages", 5.10, 1.75, -2.10, 32),
        ("CRO", "Croissant", "Bakery", 3.65, 1.40, -1.60, 25),
        ("MUF", "Blueberry Muffin", "Bakery", 3.25, 1.05, -1.30, 19),
    ]
    products = pd.DataFrame(
        [(pid, name, cat, price, cost) for pid, name, cat, price, cost, _, _ in specs],
        columns=["product_id", "product_name", "category", "base_price", "unit_cost"],
    )
    products.to_csv(data_dir / "products.csv", index=False)

    order_rows = []
    for day_idx, date in enumerate(pd.date_range("2025-01-01", periods=days)):
        units_to_sell = []
        weekday_factor = 0.92 if date.dayofweek >= 5 else 1.06
        seasonal_factor = 1 + 0.035 * day_idx / max(days - 1, 1)
        for pid, _, _, base, _, demand_elasticity, normal_units in specs:
            promo = int(rng.random() < 0.20)
            price_change = rng.choice([-0.12, -0.06, 0.00, 0.06, 0.12])
            price = round(base * (1 + price_change) * (0.94 if promo else 1), 2)
            demand = (normal_units * (price / base) ** demand_elasticity
                      * weekday_factor * seasonal_factor
                      * (1.12 if promo else 1) * rng.lognormal(0, 0.08))
            units = max(int(rng.poisson(demand)), 1)
            units_to_sell.extend([(pid, price, promo)] * units)

        rng.shuffle(units_to_sell)
        offset, order_number = 0, 0
        while offset < len(units_to_sell):
            basket_size = int(rng.choice([1, 2, 3], p=[0.65, 0.30, 0.05]))
            basket = units_to_sell[offset: offset + basket_size]
            offset += basket_size
            order_number += 1
            order_id = f"DEMO-{date:%Y%m%d}-{order_number:04d}"
            counts = {}
            for pid, price, promo in basket:
                counts[(pid, price, promo)] = counts.get((pid, price, promo), 0) + 1
            for (pid, price, promo), qty in counts.items():
                order_rows.append((date.date().isoformat(), order_id, pid, qty, price, promo))
    sales = pd.DataFrame(
        order_rows,
        columns=["date", "transaction_id", "product_id", "quantity", "unit_price", "promotion_flag"],
    )
    sales.to_csv(data_dir / "sales_transactions.csv", index=False)

    item_costs = sales.merge(products[["product_id", "category", "unit_cost"]], on="product_id")
    item_costs["cogs"] = item_costs["quantity"] * item_costs["unit_cost"]
    cat_cogs = item_costs.groupby("category")["cogs"].sum()
    inventory_rows = []
    # Explicit synthetic planning assumptions, not estimates of a real café.
    for category, assumed_days_of_stock in [("Beverages", 12), ("Bakery", 6)]:
        avg_daily_cogs = float(cat_cogs[category]) / days
        opening = round(avg_daily_cogs * assumed_days_of_stock, 2)
        closing = round(opening * float(rng.uniform(0.94, 1.06)), 2)
        shrinkage = round(float(cat_cogs[category]) * 0.005, 2)
        # Stock flow: opening + purchases - closing = sales COGS + spoilage.
        purchases = round(float(cat_cogs[category]) + closing - opening + shrinkage, 2)
        inventory_rows.append((category, opening, purchases, closing, shrinkage))
    pd.DataFrame(
        inventory_rows,
        columns=["category", "opening_inventory_value", "purchases_value",
                 "closing_inventory_value", "assumed_shrinkage_value"],
    ).to_csv(data_dir / "inventory_summary.csv", index=False)
    print("Generated 180-day synthetic POS, product and inventory data (NO real café records).")


def load_and_clean(data_dir):
    sales = pd.read_csv(data_dir / "sales_transactions.csv")
    catalog = pd.read_csv(data_dir / "products.csv")
    inventory = pd.read_csv(data_dir / "inventory_summary.csv")
    required = {
        "sales_transactions.csv": (sales, {"date", "transaction_id", "product_id", "quantity", "unit_price", "promotion_flag"}),
        "products.csv": (catalog, {"product_id", "product_name", "category", "base_price", "unit_cost"}),
        "inventory_summary.csv": (inventory, {"category", "opening_inventory_value", "purchases_value", "closing_inventory_value"}),
    }
    for filename, (frame, columns) in required.items():
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{filename}: missing columns {sorted(missing)}")

    original_count = len(sales)
    sales = sales.drop_duplicates().copy()
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    for col in ["quantity", "unit_price", "promotion_flag"]:
        sales[col] = pd.to_numeric(sales[col], errors="coerce")
    for col in ["base_price", "unit_cost"]:
        catalog[col] = pd.to_numeric(catalog[col], errors="coerce")
    for col in ["opening_inventory_value", "purchases_value", "closing_inventory_value"]:
        inventory[col] = pd.to_numeric(inventory[col], errors="coerce")
    if "assumed_shrinkage_value" in inventory.columns:
        inventory["assumed_shrinkage_value"] = pd.to_numeric(inventory["assumed_shrinkage_value"], errors="coerce")

    if catalog["product_id"].duplicated().any() or inventory["category"].duplicated().any():
        raise ValueError("product_id in catalog and category in inventory must be unique.")
    if (catalog[["base_price", "unit_cost"]].isna().any().any()
            or (catalog["base_price"] <= 0).any() or (catalog["unit_cost"] < 0).any()):
        raise ValueError("Product prices/costs are invalid.")
    inventory_values = inventory[["opening_inventory_value", "purchases_value", "closing_inventory_value"]]
    if inventory_values.isna().any().any() or (inventory_values < 0).any().any():
        raise ValueError("Inventory values must be valid nonnegative numbers.")
    valid = (sales["date"].notna() & sales["transaction_id"].notna()
             & sales["product_id"].notna() & sales["quantity"].gt(0)
             & sales["unit_price"].gt(0) & sales["promotion_flag"].isin([0, 1]))
    sales = sales.loc[valid].copy()
    if sales.empty:
        raise ValueError("No valid sales records.")
    unknown = set(sales["product_id"]) - set(catalog["product_id"])
    if unknown:
        raise ValueError(f"Unknown product IDs: {sorted(unknown)}")
    missing_cat = set(catalog["category"]) - set(inventory["category"])
    if missing_cat:
        raise ValueError(f"Inventory missing category: {sorted(missing_cat)}")

    sales = sales.merge(catalog, on="product_id", how="left", validate="many_to_one")
    sales["revenue"] = sales["quantity"] * sales["unit_price"]
    sales["cogs"] = sales["quantity"] * sales["unit_cost"]
    sales["gross_profit"] = sales["revenue"] - sales["cogs"]
    print(f"Loaded {len(sales):,} POS lines; removed {original_count - len(sales):,} invalid/duplicate lines.")
    return sales, catalog, inventory


def product_diagnostics(sales):
    table = sales.groupby(["product_id", "product_name", "category"], as_index=False).agg(
        units_sold=("quantity", "sum"), revenue=("revenue", "sum"),
        cogs=("cogs", "sum"), gross_profit=("gross_profit", "sum"),
        transactions=("transaction_id", "nunique"),
    )
    table["gross_margin_pct"] = 100 * table["gross_profit"] / table["revenue"]
    table["sales_velocity_rank"] = table["units_sold"].rank(method="min")
    table["slow_moving"] = table["units_sold"] <= table["units_sold"].quantile(0.25)
    return table.sort_values("gross_profit", ascending=False)


def inventory_diagnostics(sales, inventory):
    by_category = sales.groupby("category")["cogs"].sum().rename("sales_based_cogs")
    result = inventory.merge(by_category, on="category", how="left")
    result["inventory_usage_cost"] = (result["opening_inventory_value"]
                                       + result["purchases_value"] - result["closing_inventory_value"])
    result["average_inventory_value"] = (result["opening_inventory_value"]
                                          + result["closing_inventory_value"]) / 2
    if (result["average_inventory_value"] <= 0).any():
        raise ValueError("Average inventory value must be positive.")
    result["inventory_turnover"] = result["inventory_usage_cost"] / result["average_inventory_value"]
    result["non_sales_inventory_usage"] = result["inventory_usage_cost"] - result["sales_based_cogs"]
    if "assumed_shrinkage_value" in result.columns:
        result["unexplained_gap_after_assumed_shrinkage"] = (result["non_sales_inventory_usage"]
                                                              - result["assumed_shrinkage_value"])
    return result


def basket_analysis(sales, catalog):
    """Descriptive synthetic co-purchases; random grouping has no preferences."""
    baskets = sales.groupby("transaction_id")["product_id"].apply(set)
    counts = {}
    for basket in baskets:
        for a, b in combinations(sorted(basket), 2):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    products = catalog.set_index("product_id")
    rows = []
    for (a, b), n in counts.items():
        if products.loc[a, "category"] == products.loc[b, "category"]:
            continue
        margin = float(products.loc[a, "base_price"] - products.loc[a, "unit_cost"]
                       + products.loc[b, "base_price"] - products.loc[b, "unit_cost"])
        rows.append({"product_pair": f"{products.loc[a, 'product_name']} + {products.loc[b, 'product_name']}",
                     "co_purchase_orders": n, "combined_unit_gross_profit": margin,
                     "pair_opportunity_score": round(n * margin, 2)})
    cols = ["product_pair", "co_purchase_orders", "combined_unit_gross_profit", "pair_opportunity_score"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("pair_opportunity_score", ascending=False)


def daily_demand(sales):
    daily = sales.groupby(["date", "product_id"], as_index=False).agg(
        units_sold=("quantity", "sum"), revenue=("revenue", "sum"),
        promotion_flag=("promotion_flag", "max"),
    )
    daily["price"] = daily["revenue"] / daily["units_sold"]
    daily["weekend"] = (daily["date"].dt.dayofweek >= 5).astype(int)
    daily["day_index"] = (daily["date"] - daily["date"].min()).dt.days
    daily["log_price"] = np.log(daily["price"])
    daily["log_units"] = np.log(daily["units_sold"])
    return daily


def candidate_models():
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=2.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=90, min_samples_leaf=4, random_state=SEED, n_jobs=1
        ),
    }


def fit_log_demand(model, training):
    model.fit(training[FEATURES], training["log_units"])
    resid = training["log_units"].to_numpy() - model.predict(training[FEATURES])
    smearing = float(np.mean(np.exp(resid)))
    return model, smearing


def predict_units(model, frame, smearing):
    return np.exp(model.predict(frame[FEATURES])) * smearing


def pricing_analysis(daily, catalog):
    """Chronological 60/20/20 selection/evaluation and full-data scenarios.

    Model selection uses validation only. The final 20% test block is not used
    for fitting or selection. Full-sample refits are for illustrative scenarios.
    """
    evaluations, comparisons, recommendations, promotions, curves = [], [], [], [], {}
    fitted_for_plots = {}
    all_dates = pd.DataFrame({"date": sorted(daily["date"].unique())})
    all_dates["weekend"] = (all_dates["date"].dt.dayofweek >= 5).astype(int)
    all_dates["day_index"] = (all_dates["date"] - daily["date"].min()).dt.days

    for p in catalog.itertuples(index=False):
        sample = daily[daily["product_id"] == p.product_id].sort_values("date").copy()
        if len(sample) < 60 or sample["price"].nunique() < 4:
            raise ValueError(f"Need at least 60 days and 4 distinct prices for {p.product_name}.")
        n_train, n_train_val = int(len(sample) * 0.60), int(len(sample) * 0.80)
        train, val, test = sample.iloc[:n_train], sample.iloc[n_train:n_train_val], sample.iloc[n_train_val:]
        if train.empty or val.empty or test.empty:
            raise ValueError(f"Empty chronological train/validation/test segment for {p.product_name}.")

        candidate_errors = {}
        for name, candidate in candidate_models().items():
            fitted, adjust = fit_log_demand(candidate, train)
            val_predictions = predict_units(fitted, val, adjust)
            mae = float(mean_absolute_error(val["units_sold"], val_predictions))
            candidate_errors[name] = mae
            comparisons.append({"product_id": p.product_id, "product_name": p.product_name,
                                "candidate_model": name, "validation_days": len(val),
                                "validation_mae_units_per_day": round(mae, 3),
                                "validation_r2": round(float(r2_score(val["units_sold"], val_predictions)), 3)})
        selected_name = min(candidate_errors, key=candidate_errors.get)
        trained, adj = fit_log_demand(candidate_models()[selected_name], sample.iloc[:n_train_val])
        test_predictions = predict_units(trained, test, adj)
        test_mae = float(mean_absolute_error(test["units_sold"], test_predictions))
        test_r2 = float(r2_score(test["units_sold"], test_predictions))
        # Benchmark forecasts each future day's units at the historical mean.
        naive_level = float(sample.iloc[:n_train_val]["units_sold"].mean())
        naive_mae = float(mean_absolute_error(test["units_sold"], np.full(len(test), naive_level)))
        # Log-log OLS elasticity is interpretable even if a nonlinear forecast is selected.
        elasticity_model, elasticity_adjust = fit_log_demand(candidate_models()["Linear Regression"], sample)
        scaler = elasticity_model.named_steps["standardscaler"]
        regression = elasticity_model.named_steps["linearregression"]
        elasticity = float(regression.coef_[0] / scaler.scale_[0])
        validated = test_r2 > 0 and test_mae < naive_mae
        evaluations.append({
            "product_id": p.product_id, "product_name": p.product_name,
            "selected_forecast_model": selected_name,
            "train_days": len(train), "validation_days": len(val), "test_days": len(test),
            "validation_mae": round(candidate_errors[selected_name], 3),
            "test_mae_units_per_day": round(test_mae, 3), "test_r2": round(test_r2, 3),
            "naive_test_mae": round(naive_mae, 3),
            "estimated_price_elasticity_loglog_ols": round(elasticity, 3),
            "scenario_eligible": bool(validated),
        })

        # The separate log-log OLS model supports interpretable, smooth price
        # scenarios. The selected ML model is used for out-of-sample forecasting
        # comparisons, not for claiming causal price effects.
        full_model, full_adjust = elasticity_model, elasticity_adjust
        fitted_for_plots[p.product_name] = (full_model, full_adjust)
        regular = sample.loc[sample["promotion_flag"] == 0, "price"]
        low, high = float(regular.min()), float(regular.max())
        candidate_prices = np.linspace(low, high, 61)
        days = all_dates.copy()
        days["promotion_flag"] = 0
        def summarize_scenario(prices, promo_flags):
            scenario = days.copy()
            scenario["log_price"] = np.log(prices)
            scenario["promotion_flag"] = promo_flags
            q = predict_units(full_model, scenario, full_adjust)
            pvector = np.broadcast_to(np.asarray(prices, dtype=float), q.shape)
            return {"quantity": float(q.sum()),
                    "revenue": float(np.sum(q * pvector)),
                    "gross_profit": float(np.sum(q * (pvector - p.unit_cost)))}

        baseline = summarize_scenario(float(p.base_price), 0)
        tested = [(float(price), summarize_scenario(float(price), 0)) for price in candidate_prices]
        revenue_max_price, revenue_max = max(tested, key=lambda x: x[1]["revenue"])
        gross_max_price, gross_max = max(tested, key=lambda x: x[1]["gross_profit"])
        recommended_price = gross_max_price if validated and gross_max["gross_profit"] > baseline["gross_profit"] else float(p.base_price)
        recommended = summarize_scenario(recommended_price, 0)
        status = ("Model-supported scenario (not causal)" if validated else
                  "Hold baseline: holdout forecast did not beat naive benchmark and positive R2")
        recommendations.append({
            "product_id": p.product_id, "product_name": p.product_name,
            "selected_forecast_model": selected_name,
            "scenario_model": "Log-log Linear Regression (associational)",
            "baseline_price": round(p.base_price, 2),
            "estimated_price_elasticity": round(elasticity, 3),
            "revenue_optimal_tested_price": round(revenue_max_price, 2),
            "gross_profit_optimal_tested_price": round(gross_max_price, 2),
            "conservative_recommended_price": round(recommended_price, 2),
            "recommendation_status": status,
            "baseline_predicted_gross_profit": round(baseline["gross_profit"], 2),
            "recommended_predicted_gross_profit": round(recommended["gross_profit"], 2),
            "baseline_predicted_revenue": round(baseline["revenue"], 2),
            "recommended_predicted_revenue": round(recommended["revenue"], 2),
        })
        curves[p.product_name] = {
            "prices": candidate_prices,
            "revenues": [x[1]["revenue"] for x in tested],
            "profits": [x[1]["gross_profit"] for x in tested],
            "revenue_best_price": revenue_max_price,
            "gross_best_price": gross_max_price,
            "recommended_price": recommended_price,
            "base_price": float(p.base_price),
        }
        # Only test a small discounted promotional fraction of days;
        # treat any estimated lift as conditional on strong assumptions.
        discount_price = round(recommended_price * 0.95, 2)
        promo_days = np.arange(len(days)) % 5 == 0
        within_observed_support = float(sample["price"].min()) <= discount_price <= float(sample["price"].max())
        if validated and within_observed_support:
            proposed_prices = np.where(promo_days, discount_price, recommended_price)
            promo_outcome = summarize_scenario(proposed_prices, promo_days.astype(int))
            uplift = promo_outcome["gross_profit"] - recommended["gross_profit"]
            worthy = uplift > 0
            promo_profit, inc = round(promo_outcome["gross_profit"], 2), round(uplift, 2)
        else:
            worthy, promo_profit, inc = False, np.nan, np.nan
        promotions.append({
            "product_name": p.product_name,
            "regular_price": round(recommended_price, 2),
            "promotional_price_5pct_off": discount_price,
            "discounted_days_fraction": round(float(promo_days.mean()), 3),
            "eligible_and_within_observed_range": bool(validated and within_observed_support),
            "regular_predicted_gross_profit": round(recommended["gross_profit"], 2),
            "promotion_predicted_gross_profit": promo_profit,
            "incremental_predicted_gross_profit": inc,
            "promotional_scenario_selected": bool(worthy),
        })

    return (pd.DataFrame(evaluations), pd.DataFrame(comparisons),
            pd.DataFrame(recommendations), pd.DataFrame(promotions),
            curves, fitted_for_plots)


def save_visualizations(daily, catalog, evaluations, curves, fitted_models, output_dir):
    names = list(curves)
    names_by_id = catalog.set_index("product_id")["product_name"].to_dict()

    # Separate panels and discrete observations fix zig-zag / vertical connectors.
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        pid = next(k for k, v in names_by_id.items() if v == name)
        group = daily[daily["product_id"] == pid]
        ax.scatter(group["price"], group["units_sold"], alpha=0.22, s=18,
                   label="Daily observations")
        line_prices = np.linspace(float(group["price"].min()), float(group["price"].max()), 90)
        ref_day = int(group["day_index"].median())
        reference = pd.DataFrame({
            "log_price": np.log(line_prices), "weekend": 0,
            "promotion_flag": 0, "day_index": ref_day,
        })
        model, smear = fitted_models[name]
        ax.plot(line_prices, predict_units(model, reference, smear), lw=2,
                label="Log-log OLS fit (weekday, no promo)")
        ax.set_title(name)
        ax.set_xlabel("Price ($)")
        ax.set_ylabel("Daily units")
        ax.grid(alpha=0.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.025))
    fig.suptitle("Observed Price–Demand and Fitted Models (Synthetic Demo)", fontsize=14)
    fig.savefig(output_dir / "demand_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Legend at BOTTOM and title above panels: no overlap.
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        curve = curves[name]
        ax.plot(curve["prices"], curve["revenues"], lw=2, label="Predicted revenue")
        ax.plot(curve["prices"], curve["profits"], lw=2, label="Predicted gross profit")
        ax.axvline(curve["revenue_best_price"], ls=":", alpha=0.6, label="Best revenue tested")
        ax.axvline(curve["gross_best_price"], ls="--", alpha=0.6, label="Best gross profit tested")
        ax.set_title(name)
        ax.set_xlabel("Unit price ($)")
        ax.set_ylabel("180-day predicted total ($)")
        ax.grid(alpha=0.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.025))
    fig.suptitle("Modelled Price–Revenue and Price–Gross-Profit Scenarios (Synthetic Demo)", fontsize=14)
    fig.savefig(output_dir / "price_revenue_profit.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(8, 4.6))
    sorted_scores = evaluations.sort_values("estimated_price_elasticity_loglog_ols")
    plt.barh(sorted_scores["product_name"], sorted_scores["estimated_price_elasticity_loglog_ols"])
    plt.xlabel("Estimated price elasticity (log–log linear model)")
    plt.title("Estimated Demand Price Elasticities (Synthetic Demo)")
    plt.tight_layout()
    plt.savefig(output_dir / "price_elasticity.png", dpi=160)
    plt.close()


def save_report(products, inventory, combos, evaluations, pricing, promotions, output_dir):
    baseline = float(pricing["baseline_predicted_gross_profit"].sum())
    improved = float(pricing["recommended_predicted_gross_profit"].sum())
    promo_lift = float(promotions.loc[promotions["promotional_scenario_selected"],
                                       "incremental_predicted_gross_profit"].sum())
    combined = improved + promo_lift
    lift_pct = 100 * (combined / baseline - 1) if baseline else np.nan
    lines = [
        "CAMPUS CAFÉ PROFITABILITY ANALYSIS — SYNTHETIC DEMONSTRATION (V2)",
        "=====================================================================",
        "All POS transactions, baskets, inventory and cost assumptions are SYNTHETIC.",
        "No real university café records are represented in this demo.",
        "",
        f"Demo-generated sales gross profit (not forecast): ${products['gross_profit'].sum():,.2f}",
        f"Baseline modelled gross profit: ${baseline:,.2f}",
        f"Conservatively selected pricing scenarios: ${improved:,.2f}",
        f"Selected pricing + optional promotional scenarios: ${combined:,.2f}",
        f"Illustrative modelled change relative to baseline: {lift_pct:+.2f}%",
        "",
        "Forecast: chronological 60% train / 20% validation / 20% untouched test.",
        "Linear, Ridge and Random Forest compared on validation MAE.",
        "Price scenarios use the separate interpretable log-log OLS demand model.",
    ]
    for r in evaluations.itertuples():
        lines.append(f"  {r.product_name}: {r.selected_forecast_model}; test R2 {r.test_r2:+.3f}, "
                     f"test MAE {r.test_mae_units_per_day:.2f}, naive MAE {r.naive_test_mae:.2f}, "
                     f"log-log elasticity {r.estimated_price_elasticity_loglog_ols:+.2f}, "
                     f"scenario eligible {r.scenario_eligible}")
    lines.extend(["", "Inventory turnover for sample period (NOT annualized):"])
    for r in inventory.itertuples():
        lines.append(f"  {r.category}: {r.inventory_turnover:.2f}x; "
                     f"non-sales inventory usage ${r.non_sales_inventory_usage:,.2f}")
    lines.extend(["", "Slow-moving SKUs (bottom quartile in simulated units):"])
    for r in products[products["slow_moving"]].itertuples():
        lines.append(f"  {r.product_name}: {r.units_sold:,.0f} units; {r.gross_margin_pct:.1f}% gross margin")
    lines.extend(["", "Illustrative randomly constructed basket pairs:"])
    for r in combos.head(3).itertuples():
        lines.append(f"  {r.product_pair}: {r.co_purchase_orders} simulated co-purchase orders; "
                     f"${r.combined_unit_gross_profit:.2f} gross profit per 1+1 combination")
    lines += [
        "", "LIMITATIONS AND REPRODUCIBILITY:",
        "* All inputs generated by code; business confidentiality is NOT evidence of empirical validation.",
        "* Inventory inputs include assumed stock coverage and 0.5% synthetic shrinkage.",
        "* Simulated stock-flow accounting is constructed, not independently audited.",
        "* Demand/price elasticities are associations rather than causal estimates.",
        "* Pricing and promotion scenarios refit a separate log-log OLS model on all data.",
        "* Selection-model test scores do not validate the causal accuracy of OLS price simulations.",
        "* Weak holdout forecasts retain catalog prices rather than being auto-optimized.",
        "* Promotion results are counterfactual model estimates, not experimental evidence.",
        "* Random demo baskets do not demonstrate real co-purchasing preferences.",
        "* Gross profit excludes salaries, rent, overhead, taxes, shrinkage expenses and fixed costs.",
        "* Prices are optimized within observed non-promotion price ranges only.",
        "* No stockouts, cannibalization, dynamic competitors or cross-price elasticities modelled.",
    ]
    (output_dir / "analysis_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:13]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-demo", action="store_true",
                        help="WARNING: replace all three inputs with regenerated synthetic files")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [(DATA_DIR / file).exists() for file in INPUT_FILES]
    if args.generate_demo:
        generate_synthetic_data(DATA_DIR)
    elif not any(existing):
        generate_synthetic_data(DATA_DIR)
        print("Input files were all absent; first run created a synthetic demonstration.")
    elif not all(existing):
        raise FileNotFoundError("Incomplete data inputs. Restore all 3 CSVs or use --generate-demo (overwrites inputs).")

    sales, catalog, inventory = load_and_clean(DATA_DIR)
    product_results = product_diagnostics(sales)
    inventory_results = inventory_diagnostics(sales, inventory)
    pair_results = basket_analysis(sales, catalog)
    daily = daily_demand(sales)
    scores, comparisons, prices, promos, curves, fitted = pricing_analysis(daily, catalog)

    for filename, frame in [
        ("product_performance.csv", product_results),
        ("inventory_turnover.csv", inventory_results),
        ("high_margin_product_pairs.csv", pair_results),
        ("demand_model_evaluation.csv", scores),
        ("model_comparison.csv", comparisons),
        ("pricing_recommendations.csv", prices),
        ("promotion_scenarios.csv", promos),
    ]:
        frame.to_csv(OUTPUT_DIR / filename, index=False)
    save_visualizations(daily, catalog, scores, curves, fitted, OUTPUT_DIR)
    save_report(product_results, inventory_results, pair_results, scores, prices, promos, OUTPUT_DIR)
    print(f"\nAnalysis finished. Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
