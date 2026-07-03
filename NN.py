def build_nn_dataset(results):
    """
    Dataset pour Neural Network :
    Inputs  : t, r_t
    Output  : NPV(t, r_t)
    """

    rate_paths = results["rate_paths"]
    npv_paths = results["npv_paths"]
    time_grid = results["time_grid"]

    Nmc, n_steps = rate_paths.shape

    T_matrix = np.tile(time_grid, (Nmc, 1))

    X = np.column_stack([
        T_matrix.reshape(-1),
        rate_paths.reshape(-1)
    ])

    y = npv_paths.reshape(-1)

    return X, y


def train_neural_network_surrogate(results):
    """
    Entraîne un Neural Network pour approximer :
    V(t, r_t) ≈ NN(t, r_t)
    """

    print("\n" + "=" * 70)
    print("PARTIE 5 - MACHINE LEARNING : NEURAL NETWORK REGRESSION")
    print("=" * 70)

    X, y = build_nn_dataset(results)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    start_time = time.time()

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_split=0.20,
        epochs=100,
        batch_size=2048,
        callbacks=[early_stop],
        verbose=1
    )

    training_time = time.time() - start_time

    y_pred_scaled = model.predict(X_test_scaled, verbose=0).ravel()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nRESULTATS DU MODELE NEURAL NETWORK")
    print("=" * 70)
    print(f"Nombre total d'observations: {len(y):,}")
    print(f"Nombre d'observations train: {len(y_train):,}")
    print(f"Nombre d'observations test: {len(y_test):,}")
    print(f"Temps entraînement NN: {training_time:.4f} secondes")
    print(f"RMSE: {rmse:,.4f} EUR")
    print(f"MAE: {mae:,.4f} EUR")
    print(f"R²: {r2:.6f}")

    return {
        "model": model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "history": history,
        "X": X,
        "y": y,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "training_time": training_time
    }


def predict_npv_paths_nn(results, nn_results):
    """
    Prédit toute la matrice NPV avec le Neural Network.
    """

    rate_paths = results["rate_paths"]
    time_grid = results["time_grid"]

    Nmc, n_steps = rate_paths.shape
    T_matrix = np.tile(time_grid, (Nmc, 1))

    X_full = np.column_stack([
        T_matrix.reshape(-1),
        rate_paths.reshape(-1)
    ])

    X_full_scaled = nn_results["scaler_X"].transform(X_full)

    start_time = time.time()

    y_pred_scaled = nn_results["model"].predict(
        X_full_scaled,
        batch_size=4096,
        verbose=0
    ).ravel()

    prediction_time = time.time() - start_time

    y_pred_full = nn_results["scaler_y"].inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    npv_nn_paths = y_pred_full.reshape(Nmc, n_steps)

    print(f"Temps prédiction NN complet: {prediction_time:.4f} secondes")

    return npv_nn_paths, prediction_time


def run_nn_cva_analysis(results):
    """
    Calcule la CVA à partir des NPV prédites par Neural Network.
    """

    nn_results = train_neural_network_surrogate(results)

    npv_nn_paths, prediction_time = predict_npv_paths_nn(results, nn_results)

    market_data = MarketData(
        r=0.02,
        sigma=0.025,
        initial_rate=0.02,
        theta=0.04,
        spread_credit=0.015,
        recovery_rate=0.4,
        kappa=0.3
    )

    cva_engine = CVAEngine(market_data)

    lambda_cp = market_data.spread_credit / (1 - market_data.recovery_rate)

    cva_nn = cva_engine.calculate_cva_formula(
        npv_nn_paths,
        results["time_grid"],
        results["rate_paths"],
        lambda_cp
    )

    cva_mc = results["cva_no_wwr"]

    abs_error = abs(cva_nn - cva_mc)
    rel_error = abs_error / abs(cva_mc)

    exposure_nn = cva_engine.calculate_exposure_metrics(
        npv_nn_paths,
        results["time_grid"]
    )

    print("\n" + "=" * 70)
    print("RESULTATS CVA AVEC NEURAL NETWORK")
    print("=" * 70)

    print(f"CVA Monte Carlo classique: {cva_mc:,.4f} EUR")
    print(f"CVA Neural Network: {cva_nn:,.4f} EUR")
    print(f"Erreur absolue: {abs_error:,.4f} EUR")
    print(f"Erreur relative: {rel_error * 100:.6f}%")
    print(f"EPE Neural Network: {exposure_nn['epe']:,.4f} EUR")
    print(f"Max PFE 95% Neural Network: {exposure_nn['max_pfe']:,.4f} EUR")

    nn_results.update({
        "npv_nn_paths": npv_nn_paths,
        "prediction_time": prediction_time,
        "cva_nn": cva_nn,
        "abs_error_cva": abs_error,
        "rel_error_cva": rel_error,
        "exposure_nn": exposure_nn
    })

    return nn_results


def plot_nn_results(results, nn_results):
    """
    Graphiques de validation Neural Network.
    """

    time_grid = results["time_grid"]

    ee_mc = results["exposure_metrics"]["ee"]
    ee_nn = nn_results["exposure_nn"]["ee"]

    plt.figure(figsize=(12, 5))
    plt.plot(time_grid, ee_mc, label="EE Monte Carlo", linewidth=2)
    plt.plot(time_grid, ee_nn, "--", label="EE Neural Network", linewidth=2)
    plt.title("Expected Exposure : Monte Carlo vs Neural Network")
    plt.xlabel("Temps (années)")
    plt.ylabel("Expected Exposure (EUR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(
        nn_results["y_test"],
        nn_results["y_pred"],
        alpha=0.25,
        s=5
    )

    min_val = min(np.min(nn_results["y_test"]), np.min(nn_results["y_pred"]))
    max_val = max(np.max(nn_results["y_test"]), np.max(nn_results["y_pred"]))

    plt.plot([min_val, max_val], [min_val, max_val], linewidth=2)
    plt.title("NPV réelle vs NPV prédite par Neural Network")
    plt.xlabel("NPV réelle")
    plt.ylabel("NPV prédite")
    plt.grid(True, alpha=0.3)
    plt.show()

    errors = nn_results["y_pred"] - nn_results["y_test"]

    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=60, alpha=0.7, edgecolor="black")
    plt.title("Distribution des erreurs de prédiction Neural Network")
    plt.xlabel("Erreur de prédiction")
    plt.ylabel("Fréquence")
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(nn_results["history"].history["loss"], label="Training Loss")
    plt.plot(nn_results["history"].history["val_loss"], label="Validation Loss")
    plt.title("Courbe d'apprentissage Neural Network")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

nn_results = run_nn_cva_analysis(results)
plot_nn_results(results, nn_results)

print("\n" + "=" * 90)
print("COMPARAISON AVEC NEURAL NETWORK")
print("=" * 90)

print(f"Monte Carlo classique | CVA = {results['cva_no_wwr']:,.4f} EUR | Erreur = 0.0000%")

print(
    f"Neural Network | CVA = {nn_results['cva_nn']:,.4f} EUR "
    f"| Erreur = {nn_results['rel_error_cva'] * 100:.6f}% "
    f"| RMSE = {nn_results['rmse']:,.4f} EUR "
    f"| MAE = {nn_results['mae']:,.4f} EUR "
    f"| R² = {nn_results['r2']:.6f} "
    f"| Train = {nn_results['training_time']:.4f}s "
    f"| Predict = {nn_results['prediction_time']:.4f}s"
)