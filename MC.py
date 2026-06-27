@dataclass
class MarketData:
    """Données de marché pour le pricing"""
    r: float  # taux sans risque
    sigma: float  # volatilité

    initial_rate: float
    spread_credit: float
    recovery_rate: float
    kappa: float = 0.1
    theta: float = 0.03


@dataclass
class SwapParameters:
    """Paramètres du swap"""
    notional: float
    maturity: float
    fixed_rate: float
    payment_frequency: int
    is_payer: bool


class VasicekModel:
    """Modèle de Vasicek avec paramètres alignés au rapport"""

    def __init__(self, r0: float, kappa: float, theta: float, sigma: float):
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def simulate_paths(self, T: float, dt: float, Nmc: int) -> np.ndarray:
        """Simule les trajectoires de taux"""
        n_steps = int(round(T / dt))
        rate_paths = np.zeros((Nmc, n_steps + 1))
        rate_paths[:, 0] = self.r0

        # Génération vectorisée pour plus d'efficacité
        dW = np.random.normal(0, np.sqrt(dt), (Nmc, n_steps))

        for i in range(n_steps):
            drift = self.kappa * (self.theta - rate_paths[:, i]) * dt
            diffusion = self.sigma * dW[:, i]
            rate_paths[:, i + 1] = rate_paths[:, i] + drift + diffusion

            # Limite inférieure pour éviter les taux trop négatifs
            rate_paths[:, i + 1] = np.maximum(rate_paths[:, i + 1], -0.05)

        return rate_paths

    def zero_coupon_bond(self, tau: float, r_current: np.ndarray) -> np.ndarray:
        """Prix du zéro-coupon Vasicek"""
        if tau <= 0:
            return np.ones_like(r_current)

        if self.kappa == 0:
            B = tau
            A = np.exp(-self.theta * tau + (self.sigma**2 * tau**3) / 6)
        else:
            B = (1 - np.exp(-self.kappa * tau)) / self.kappa
            term1 = (self.theta - self.sigma**2 / (2 * self.kappa**2)) * (B - tau)
            term2 = (self.sigma**2 * B**2) / (4 * self.kappa)
            A = np.exp(term1 - term2)

        return A * np.exp(-B * r_current)

class InterestRateSwap:
    """Swap de taux avec valorisation complète"""

    def __init__(self, params: SwapParameters, market_data: MarketData,
                 rate_model: VasicekModel):
        self.params = params
        self.market_data = market_data
        self.rate_model = rate_model
        self.payment_dates = self._generate_payment_dates()

        if self.params.fixed_rate == 0.0:
            self.params.fixed_rate = self._calculate_atm_rate()

    def _generate_payment_dates(self) -> np.ndarray:
        dt = 1.0 / self.params.payment_frequency
        return np.arange(dt, self.params.maturity + dt, dt)

    def _calculate_atm_rate(self) -> float:
        """Calcul du taux swap ATM"""
        dt = 1.0 / self.params.payment_frequency

        zcb_prices = []
        for date in self.payment_dates:
            price = self.rate_model.zero_coupon_bond(
                date, np.array([self.market_data.initial_rate])
            )[0]
            zcb_prices.append(price)

        zcb_prices = np.array(zcb_prices)
        annuity = np.sum(dt * zcb_prices)

        if annuity > 0:
            return (1 - zcb_prices[-1]) / annuity
        else:
            return self.market_data.initial_rate

    def calculate_npv_paths(self, rate_paths: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        """Calcul de la NPV par re-valorisation complète"""
        Nmc, n_steps = rate_paths.shape
        npv_paths = np.zeros_like(rate_paths)
        dt_payment = 1.0 / self.params.payment_frequency

        for i in range(n_steps - 1):
            t = time_grid[i]
            r_t = rate_paths[:, i]

            future_payment_dates = self.payment_dates[self.payment_dates > t]

            if len(future_payment_dates) == 0:
                npv_paths[:, i] = 0.0
                continue

            # Jambe fixe
            pv_fixed_leg = np.zeros(Nmc)
            for payment_date in future_payment_dates:
                tau = payment_date - t
                if tau > 0:
                    zc_prices = self.rate_model.zero_coupon_bond(tau, r_t)
                    pv_fixed_leg += zc_prices * dt_payment

            pv_fixed_leg *= self.params.fixed_rate * self.params.notional

            # Jambe flottante
            maturity_tau = self.params.maturity - t
            if maturity_tau > 0:
                zc_maturity = self.rate_model.zero_coupon_bond(maturity_tau, r_t)
                pv_float_leg = self.params.notional * (1 - zc_maturity)
            else:
                pv_float_leg = np.zeros(Nmc)

            # NPV selon position
            if self.params.is_payer:
                npv_paths[:, i] = pv_float_leg - pv_fixed_leg
            else:
                npv_paths[:, i] = pv_fixed_leg - pv_float_leg

        return npv_paths

class DefaultModel:
    """Modèle de défaut avec intensité constante"""

    def __init__(self, lambda_default: float, recovery_rate: float):
        self.lambda_default = lambda_default
        self.recovery_rate = recovery_rate

    def simulate_default_times(self, T: float, Nmc: int) -> np.ndarray:
        """
        Simule les temps de défaut avec une intensité constante λ
        """
        U = np.random.uniform(0, 1, Nmc)

        # Temps de défaut ~ Exponentielle(λ)
        return -np.log(U) / self.lambda_default

class CVAEngine:
    """Moteur CVA avec actualisation stochastique"""

    def __init__(self, market_data: MarketData):
        self.market_data = market_data

    def calculate_exposure_metrics(self, npv_paths: np.ndarray, time_grid: np.ndarray) -> dict:
        """Calcule les métriques d'exposition"""
        positive_exposure = np.maximum(npv_paths, 0)

        ee = np.mean(positive_exposure, axis=0)
        pfe_95 = np.percentile(positive_exposure, 95, axis=0)
        pfe_99 = np.percentile(positive_exposure, 99, axis=0)
        epe = np.mean(ee)

        return {
            'ee': ee,
            'pfe_95': pfe_95,
            'pfe_99': pfe_99,
            'epe': epe,
            'max_pfe': np.max(pfe_95)
        }

    def calculate_stochastic_discount_factors(self, rate_paths: np.ndarray,
                                            time_grid: np.ndarray) -> np.ndarray:
        """Calcul des facteurs d'actualisation stochastiques"""
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 0.0

        # Ajustement des dimensions si nécessaire
        n_rate_steps = rate_paths.shape[1]
        n_time_steps = len(time_grid)
        rates_for_integration = rate_paths[:, :min(n_rate_steps, n_time_steps)]

        # Intégration trapézoïdale
        integrated_rates = np.cumsum(rates_for_integration * dt, axis=1)

        return np.exp(-integrated_rates)

    def calculate_cva_formula(self, npv_paths, time_grid, rate_paths, lambda_default):
      positive_exposure = np.maximum(npv_paths, 0)
      epe = np.mean(positive_exposure, axis=0)

      discount_factors = self.calculate_stochastic_discount_factors(rate_paths, time_grid)
      avg_discount = np.mean(discount_factors, axis=0)

      lgd = 1 - self.market_data.recovery_rate

      cva = 0.0

      for i in range(1, len(time_grid)):
          pd_i = np.exp(-lambda_default * time_grid[i-1]) - np.exp(-lambda_default * time_grid[i])

          term = 0.5 * (
              avg_discount[i-1] * epe[i-1]
              + avg_discount[i] * epe[i]
          )

          cva += term * pd_i

      return lgd * cva

def run_cva_analysis() -> Dict:
    """Fonction principale d'analyse CVA avec tous les tests de validation"""

    print("ANALYSE CVA - VALIDATION ET VERIFICATION")
    print("="*70)

    # Paramètres alignés avec le rapport
    Nmc = 20000
    T = 5.0
    dt = 1/12

    # PARAMETRES CORRIGES selon le rapport
    market_data = MarketData(
        r=0.02,
        sigma=0.025,         # 2.5% comme dans le rapport
        initial_rate=0.02,
        theta=0.04,
        spread_credit=0.015,
        recovery_rate=0.4,
        kappa=0.3            # 0.3 comme dans le rapport
    )

    swap_params = SwapParameters(
        notional=1_000_000,
        maturity=T,
        fixed_rate=0.0,
        payment_frequency=4,
        is_payer=True
    )

    print("CONFIGURATION DU MODELE:")
    print(f"  Taux initial r0: {market_data.initial_rate:.2%}")
    print(f"  Theta (LT mean): {market_data.theta:.2%}")
    print(f"  Kappa (mean rev): {market_data.kappa:.2f}")
    print(f"  Sigma (vol): {market_data.sigma:.3%}")
    print(f"  Spread credit: {market_data.spread_credit:.3%}")
    print(f"  Recovery rate: {market_data.recovery_rate:.1%}")

    # 1. Initialisation
    print("\n1. INITIALISATION DES MODELES...")
    start_time = time.time()

    rate_model = VasicekModel(
        market_data.initial_rate, market_data.kappa,
        market_data.theta, market_data.sigma
    )

    # Test: Vérification de la courbe des taux
    zcb_1y = rate_model.zero_coupon_bond(1.0, np.array([market_data.initial_rate]))[0]
    zcb_5y = rate_model.zero_coupon_bond(5.0, np.array([market_data.initial_rate]))[0]
    print(f"  P(0,1Y) = {zcb_1y:.4f}")
    print(f"  P(0,5Y) = {zcb_5y:.4f}")
    print(f"  Courbe: {'Ascendante ✓' if zcb_1y > zcb_5y else 'Descendante'}")

    # 2. Swap ATM
    swap = InterestRateSwap(swap_params, market_data, rate_model)
    print(f"\n2. SWAP ATM:")
    print(f"  Taux fixe ATM: {swap.params.fixed_rate:.4%}")

    # 3. Simulation
    print(f"\n3. SIMULATION MONTE CARLO ({Nmc:,} trajectoires)...")
    n_steps = int(round(T / dt))
    time_grid = np.linspace(0, T, n_steps + 1)
    rate_paths = rate_model.simulate_paths(T, dt, Nmc)

    # Test: Convergence vers theta
    taux_final_moyen = np.mean(rate_paths[:, -1])
    print(f"  Taux final moyen: {taux_final_moyen:.3%}")
    print(f"  Convergence test: {'✓' if abs(taux_final_moyen - market_data.theta) < 0.01 else '✗'}")

    # 4. Calcul NPV
    print("\n4. CALCUL DES EXPOSITIONS...")
    npv_paths = swap.calculate_npv_paths(rate_paths, time_grid)

    # Test: Swap ATM
    npv_initiale = np.mean(npv_paths[:, 0])
    print(f"  NPV initiale moyenne: {npv_initiale:,.0f} EUR")
    print(f"  Test ATM: {'✓' if abs(npv_initiale) < 1000 else '✗'}")

    # 5. Métriques d'exposition
    cva_engine = CVAEngine(market_data)
    exposure_metrics = cva_engine.calculate_exposure_metrics(npv_paths, time_grid)
    print(f"  EPE: {exposure_metrics['epe']:,.0f} EUR")
    print(f"  Max PFE 95%: {exposure_metrics['max_pfe']:,.0f} EUR")

    # 6. Modèle de défaut
    lambda_cp = market_data.spread_credit / (1 - market_data.recovery_rate)
    default_model = DefaultModel(lambda_cp, market_data.recovery_rate)
    print(f"\n5. MODELE DE DEFAUT:")
    print(f"  Lambda: {lambda_cp:.4f}")
    print(f"  Prob survie 5Y: {np.exp(-lambda_cp * T):.2%}")

    # 7. CVA sans WWR
    print("\n6. CALCUL CVA SANS WWR...")
    cva_no_wwr = cva_engine.calculate_cva_formula(npv_paths,time_grid,rate_paths,lambda_cp)

    print("RESULTATS FINAUX")

    print("VALIDATION vs RAPPORT")

    target_cva_no_wwr = 1050

    print(f"CVA sans WWR - Cible: {target_cva_no_wwr} EUR, Obtenu: {cva_no_wwr:.0f} EUR")
    print(f"Écart: {abs(cva_no_wwr - target_cva_no_wwr):.0f} EUR")

    # Tests de validation
    print("\n" + "="*70)
    print("TESTS DE VALIDATION")
    print("="*70)

    tests_passed = 0
    tests_total = 2

    # Test 1: Convergence Vasicek
    test1 = abs(taux_final_moyen - market_data.theta) < 0.01
    print(f"1. Convergence Vasicek: {'✓ PASSE' if test1 else '✗ ECHOUE'}")
    if test1: tests_passed += 1

    # Test 2: Swap ATM
    test2 = abs(npv_initiale) < 1000
    print(f"2. Swap ATM (NPV~0): {'✓ PASSE' if test2 else '✗ ECHOUE'}")
    if test2: tests_passed += 1

    print(f"\nRésultat: {tests_passed}/{tests_total} tests passés")

    elapsed_time = time.time() - start_time
    print(f"\nTemps d'exécution: {elapsed_time:.1f} secondes")

    # Retour des résultats pour analyse ultérieure
    return {
    'cva_no_wwr': cva_no_wwr,
    'exposure_metrics': exposure_metrics,
    'rate_paths': rate_paths,
    'npv_paths': npv_paths,
    'time_grid': time_grid
}

def plot_results(results):

    fig, axes = plt.subplots(2, 2, figsize=(15,10))

    time_grid = results['time_grid']

    # STEP 1 : Market Data Projections
    ax1 = axes[0,0]

    for i in range(min(100, results['rate_paths'].shape[0])):
        ax1.plot(time_grid,
                 results['rate_paths'][i,:],
                 alpha=0.15)

    ax1.set_title("Step 1 - Market Data Projections")
    ax1.grid(True)

    # STEP 2 : MtM
    ax2 = axes[0,1]

    for i in range(min(100, results['npv_paths'].shape[0])):
        ax2.plot(time_grid,
                 results['npv_paths'][i,:],
                 alpha=0.15)

    ax2.axhline(0,color='black')
    ax2.set_title("Step 2 - MtM Projections")
    ax2.grid(True)

    # STEP 3 : Positive Exposures
    ax3 = axes[1,0]

    positive_exposure = np.maximum(results['npv_paths'],0)

    for i in range(min(100, positive_exposure.shape[0])):
        ax3.plot(time_grid,
                 positive_exposure[i,:],
                 alpha=0.15)

    ax3.set_title("Step 3 - Exposure Projections")
    ax3.grid(True)

    # STEP 4 : EE/PFE
    ax4 = axes[1,1]

    ee = results['exposure_metrics']['ee']
    pfe95 = results['exposure_metrics']['pfe_95']
    pfe99 = results['exposure_metrics']['pfe_99']

    ax4.plot(time_grid[:len(ee)], ee, label="EE")
    ax4.plot(time_grid[:len(pfe95)], pfe95, label="PFE 95%")
    ax4.plot(time_grid[:len(pfe99)], pfe99, label="PFE 99%")

    ax4.legend()
    ax4.set_title("Step 4 - Exposure Indicators")
    ax4.grid(True)

    plt.suptitle(" ",
                 fontsize=16,
                 fontweight='bold')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    start_time = time.time()
    # Exécution principale
    results = run_cva_analysis()

    # Génération des graphiques
    plot_results(results)

    end_time = time.time()

    print(f"\nTotal execution time: {end_time - start_time:.4f} seconds")
    print("\nAnalyse terminée avec succès!")