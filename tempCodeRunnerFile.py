# --- Prediction and Visualization ---
if os.path.exists(model_path):
    model = load_model(model_path)

    # Prepare test data
    test_start = '2023-01-01'
    test_data = yf.download(stock_symbol, start=test_start, end=today)
    test_data.reset_index(inplace=True)
    actual_prices = test_data['Close'].values

    dataset_total = pd.concat((data['Close'], test_data['Close']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(test_data) - 100:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    x_test = []
    for i in range(100, inputs.shape[0]):
        x_test.append(inputs[i-100:i])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot actual vs. predicted prices
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=test_data['Date'], y=actual_prices, mode='lines', name='Actual Price'))
    fig_pred.add_trace(go.Scatter(x=test_data['Date'], y=predicted_prices.flatten(), mode='lines', name='Predicted Price'))

    st.plotly_chart(fig_pred)
