def simulate_vasicek_paths_sobol(rate_model, T, dt, Nqmc, scramble=True, seed=123):
    """
    Simulation Vasicek avec séquence Sobol.
    Remplace les normales pseudo-aléatoires par des normales issues de Sobol.
    """

    n_steps = int(round(T / dt))
    rate_paths = np.zeros((Nqmc, n_steps + 1))
    rate_paths[:, 0] = rate_model.r0

    sampler = qmc.Sobol(d=n_steps, scramble=scramble, seed=seed)

    # Sobol marche mieux avec une puissance de 2
    U = sampler.random_base2(m=int(np.log2(Nqmc)))

    # éviter 0 et 1 pour norm.ppf
    eps = 1e-12
    U = np.clip(U, eps, 1 - eps)

    Z = norm.ppf(U)
    dW = np.sqrt(dt) * Z

    for i in range(n_steps):
        drift = rate_model.kappa * (rate_model.theta - rate_paths[:, i]) * dt
        diffusion = rate_model.sigma * dW[:, i]
        rate_paths[:, i + 1] = rate_paths[:, i] + drift + diffusion
        rate_paths[:, i + 1] = np.maximum(rate_paths[:, i + 1], -0.05)

    return rate_paths


def run_qmc_cva_analysis(Nqmc=4096, seed=123):
    """
    Partie 3 : CVA avec Quasi-Monte Carlo Sobol.
    """

    print("\n" + "=" * 70)
    print("PARTIE 3 - QUASI MONTE CARLO SOBOL")
    print("=" * 70)

    T = 5.0
    dt = 1 / 12

    market_data = MarketData(
        r=0.02,
        sigma=0.025,
        initial_rate=0.02,
        theta=0.04,
        spread_credit=0.015,
        recovery_rate=0.4,
        kappa=0.3
    )

    swap_params = SwapParameters(
        notional=1_000_000,
        maturity=T,
        fixed_rate=0.0,
        payment_frequency=4,
        is_payer=True
    )

    rate_model = VasicekModel(
        market_data.initial_rate,
        market_data.kappa,
        market_data.theta,
        market_data.sigma
    )

    swap = InterestRateSwap(swap_params, market_data, rate_model)
    cva_engine = CVAEngine(market_data)

    lambda_cp = market_data.spread_credit / (1 - market_data.recovery_rate)

    n_steps = int(round(T / dt))
    time_grid = np.linspace(0, T, n_steps + 1)

    start_time = time.time()

    rate_paths_qmc = simulate_vasicek_paths_sobol(
        rate_model=rate_model,
        T=T,
        dt=dt,
        Nqmc=Nqmc,
        scramble=True,
        seed=seed
    )

    npv_paths_qmc = swap.calculate_npv_paths(rate_paths_qmc, time_grid)

    cva_qmc = cva_engine.calculate_cva_formula(
        npv_paths_qmc,
        time_grid,
        rate_paths_qmc,
        lambda_cp
    )

    exposure_qmc = cva_engine.calculate_exposure_metrics(npv_paths_qmc, time_grid)

    elapsed_time = time.time() - start_time

    print(f"Nombre de trajectoires Sobol: {Nqmc:,}")
    print(f"CVA QMC Sobol: {cva_qmc:,.2f} EUR")
    print(f"EPE QMC: {exposure_qmc['epe']:,.2f} EUR")
    print(f"Max PFE 95% QMC: {exposure_qmc['max_pfe']:,.2f} EUR")
    print(f"Temps QMC: {elapsed_time:.4f} secondes")

    return {
        "cva_qmc": cva_qmc,
        "rate_paths_qmc": rate_paths_qmc,
        "npv_paths_qmc": npv_paths_qmc,
        "exposure_qmc": exposure_qmc,
        "time_grid": time_grid,
        "time_qmc": elapsed_time,
        "Nqmc": Nqmc
    }


def compare_qmc_with_mc(results_mc, qmc_sizes=[1024, 2048, 4096, 8192]):
    """
    Compare Monte Carlo classique avec plusieurs tailles QMC.
    Attention : Sobol nécessite des puissances de 2.
    """

    comparison = []

    cva_mc = results_mc["cva_no_wwr"]

    for Nqmc in qmc_sizes:
        res_qmc = run_qmc_cva_analysis(Nqmc=Nqmc)

        abs_error = abs(res_qmc["cva_qmc"] - cva_mc)
        rel_error = abs_error / abs(cva_mc)

        comparison.append({
            "Nqmc": Nqmc,
            "CVA_QMC": res_qmc["cva_qmc"],
            "Abs_Error": abs_error,
            "Rel_Error": rel_error,
            "Time": res_qmc["time_qmc"]
        })

    print("\n" + "=" * 80)
    print("COMPARAISON MONTE CARLO VS QUASI MONTE CARLO")
    print("=" * 80)

    print(f"CVA Monte Carlo référence: {cva_mc:,.4f} EUR")

    for row in comparison:
        print(
            f"Nqmc={row['Nqmc']:,} | "
            f"CVA={row['CVA_QMC']:.4f} | "
            f"Erreur={row['Abs_Error']:.4f} | "
            f"Erreur %={row['Rel_Error'] * 100:.6f}% | "
            f"Temps={row['Time']:.4f}s"
        )

    return comparison

comparison_qmc = compare_qmc_with_mc(
    results_mc=results,
    qmc_sizes=[1024, 2048, 4096, 8192]
)