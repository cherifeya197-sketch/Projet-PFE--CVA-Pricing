def price_swap_at_time_on_rates(swap, t, r_values):
    """
    Valorisation exacte du swap à une date t pour plusieurs valeurs de taux r_values.
    Utilise la même logique que calculate_npv_paths, mais sur des noeuds d'interpolation.
    """
    r_values = np.asarray(r_values)
    N = len(r_values)
    dt_payment = 1.0 / swap.params.payment_frequency

    future_payment_dates = swap.payment_dates[swap.payment_dates > t]

    if len(future_payment_dates) == 0:
        return np.zeros(N)

    pv_fixed_leg = np.zeros(N)

    for payment_date in future_payment_dates:
        tau = payment_date - t
        if tau > 0:
            zc_prices = swap.rate_model.zero_coupon_bond(tau, r_values)
            pv_fixed_leg += zc_prices * dt_payment

    pv_fixed_leg *= swap.params.fixed_rate * swap.params.notional

    maturity_tau = swap.params.maturity - t

    if maturity_tau > 0:
        zc_maturity = swap.rate_model.zero_coupon_bond(maturity_tau, r_values)
        pv_float_leg = swap.params.notional * (1 - zc_maturity)
    else:
        pv_float_leg = np.zeros(N)

    if swap.params.is_payer:
        return pv_float_leg - pv_fixed_leg
    else:
        return pv_fixed_leg - pv_float_leg


def calculate_npv_paths_polynomial_approximation(
    swap,
    rate_paths,
    time_grid,
    n_nodes=7,
    q_low=1,
    q_high=99
):
    """
    Approximation polynomiale de la NPV :
    V(t, r_t) ≈ g(t, r_t)

    À chaque date t :
    - on prend quelques noeuds de taux
    - on calcule la vraie NPV seulement sur ces noeuds
    - on interpole avec un polynôme
    - on évalue ce polynôme sur toutes les trajectoires Monte Carlo
    """

    Nmc, n_steps = rate_paths.shape
    npv_approx_paths = np.zeros_like(rate_paths)

    exact_valuations_count = 0

    for i in range(n_steps - 1):
        t = time_grid[i]
        r_t = rate_paths[:, i]

        if t >= swap.params.maturity:
            npv_approx_paths[:, i] = 0.0
            continue

        r_min = np.percentile(r_t, q_low)
        r_max = np.percentile(r_t, q_high)

        nodes = np.linspace(r_min, r_max, n_nodes)

        V_nodes = price_swap_at_time_on_rates(swap, t, nodes)
        exact_valuations_count += n_nodes

        interpolator = BarycentricInterpolator(nodes, V_nodes)

        npv_approx_paths[:, i] = interpolator(r_t)

    return npv_approx_paths, exact_valuations_count


def run_polynomial_optimization(results_mc, n_nodes=7):
    """
    Partie 2 : approximation polynomiale.
    On ne recalcule pas la CVA Monte Carlo.
    On calcule seulement la CVA polynomiale à partir des NPV approximées.
    """

    print("\n" + "=" * 70)
    print("PARTIE 2 - OPTIMISATION PAR APPROXIMATION POLYNOMIALE")
    print("=" * 70)

    rate_paths = results_mc["rate_paths"]
    time_grid = results_mc["time_grid"]

    T = time_grid[-1]

    market_data = MarketData(
        r=0.02,
        sigma=0.025,
        initial_rate=0.02,
        theta=0.04,
        spread_credit=0.015,
        recovery_rate=0.4,
        kappa=0.3
    )

    rate_model = VasicekModel(
        market_data.initial_rate,
        market_data.kappa,
        market_data.theta,
        market_data.sigma
    )

    swap_params = SwapParameters(
        notional=1_000_000,
        maturity=T,
        fixed_rate=0.0,
        payment_frequency=4,
        is_payer=True
    )

    swap = InterestRateSwap(swap_params, market_data, rate_model)

    cva_engine = CVAEngine(market_data)
    lambda_cp = market_data.spread_credit / (1 - market_data.recovery_rate)

    start_time = time.time()

    npv_poly_paths, exact_valuations_count = calculate_npv_paths_polynomial_approximation(
        swap=swap,
        rate_paths=rate_paths,
        time_grid=time_grid,
        n_nodes=n_nodes
    )

    cva_poly = cva_engine.calculate_cva_formula(
        npv_poly_paths,
        time_grid,
        rate_paths,
        lambda_cp
    )

    elapsed_time = time.time() - start_time

    exposure_poly = cva_engine.calculate_exposure_metrics(npv_poly_paths, time_grid)

    cva_mc = results_mc["cva_no_wwr"]
    abs_error = abs(cva_poly - cva_mc)
    rel_error = abs_error / abs(cva_mc)

    print(f"Nombre de noeuds d'interpolation: {n_nodes}")
    print(f"Nombre de valorisations exactes utilisées: {exact_valuations_count:,}")
    print(f"CVA Monte Carlo déjà calculée: {cva_mc:,.2f} EUR")
    print(f"CVA approximation polynomiale: {cva_poly:,.2f} EUR")
    print(f"Erreur absolue: {abs_error:,.2f} EUR")
    print(f"Erreur relative: {rel_error:.4%}")
    print(f"EPE approximation polynomiale: {exposure_poly['epe']:,.2f} EUR")
    print(f"Max PFE 95% approximation polynomiale: {exposure_poly['max_pfe']:,.2f} EUR")
    print(f"Temps optimisation: {elapsed_time:.4f} secondes")

    return {
        "cva_poly": cva_poly,
        "npv_poly_paths": npv_poly_paths,
        "exposure_poly": exposure_poly,
        "exact_valuations_count": exact_valuations_count,
        "time_poly": elapsed_time,
        "abs_error": abs_error,
        "rel_error": rel_error
    }

def plot_polynomial_comparison(results_mc, results_poly):
    """
    Graphiques de comparaison entre Monte Carlo classique et approximation polynomiale.
    """

    time_grid = results_mc["time_grid"]

    ee_mc = results_mc["exposure_metrics"]["ee"]
    ee_poly = results_poly["exposure_poly"]["ee"]

    pfe_mc = results_mc["exposure_metrics"]["pfe_95"]
    pfe_poly = results_poly["exposure_poly"]["pfe_95"]

    plt.figure(figsize=(14, 6))

    plt.plot(time_grid, ee_mc, label="EE Monte Carlo classique", linewidth=2)
    plt.plot(time_grid, ee_poly, "--", label="EE Approximation polynomiale", linewidth=2)

    plt.plot(time_grid, pfe_mc, label="PFE 95% Monte Carlo classique", linewidth=2)
    plt.plot(time_grid, pfe_poly, "--", label="PFE 95% Approximation polynomiale", linewidth=2)

    plt.title("Comparison of Exposure Profiles")
    plt.xlabel("Time (Years)")
    plt.ylabel("Exposure (EUR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    error_ee = np.abs(ee_poly - ee_mc)

    plt.figure(figsize=(12, 5))
    plt.plot(time_grid, error_ee, linewidth=2)
    plt.title("Absolute Error of the Expected Exposure")
    plt.xlabel("Time (Years)")
    plt.ylabel(r"$|EE_{\mathrm{Approximation}} - EE_{\mathrm{Monte\ Carlo}}|$")
    plt.grid(True, alpha=0.3)
    plt.show()


results_poly = run_polynomial_optimization(results, n_nodes=7)
plot_polynomial_comparison(results, results_poly)

def compare_polynomial_nodes(results, nodes_list=[3, 5, 7, 9]):
    comparison = []

    for n in nodes_list:
        res_poly = run_polynomial_optimization(results, n_nodes=n)

        comparison.append({
            "Nodes": n,
            "EPE": res_poly["exposure_poly"]["epe"],
            "Max PFE 95": res_poly["exposure_poly"]["max_pfe"],
            "Exact Valuations": res_poly["exact_valuations_count"],
            "Time (s)": res_poly["time_poly"]
        })

    print("\nCOMPARAISON DES APPROXIMATIONS POLYNOMIALES")
    print("=" * 80)

    for row in comparison:
        print(
            f"N={row['Nodes']} | "
            f"EPE={row['EPE']:.2f} EUR | "
            f"Max PFE 95%={row['Max PFE 95']:.2f} EUR | "
            f"Valuations={row['Exact Valuations']} | "
            f"Temps={row['Time (s)']:.4f}s"
        )

    return comparison

comparison_poly = compare_polynomial_nodes(results, nodes_list=[2, 3, 5, 7, 9])