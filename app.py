from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import io
import base64
import matplotlib.dates as mdates

app = Flask(__name__)
try:
    data = pd.read_csv('Historical-Product-Demand.csv')
    # Convert 'Order_Demand' to numeric, handling potential errors
    data['Order_Demand'] = pd.to_numeric(data['Order_Demand'], errors='coerce')
    data.dropna(subset=['Order_Demand'], inplace=True)
    data['Order_Demand'] = data['Order_Demand'].astype(int)
except FileNotFoundError:
    print("Error: 'Historical-Product-Demand.csv' not found. Place the file in the same directory as the script.")
    exit()
except Exception as e:
    print(f"Error loading or processing the data file: {e}")
    exit()
def prepare_data(product_code):
    product_data = data[data['Product_Code'] == product_code].copy()

    if product_data.empty:
        return pd.Series()
    try:
        product_data.loc[:, 'Date'] = pd.to_datetime(product_data['Date'], errors='coerce')
        product_data.set_index('Date', inplace=True)
        # Handle missing values by forward filling, resample yearly here
        order_demand = product_data['Order_Demand'].resample('Y').sum().fillna(method='ffill')
        #Filter the year to the past and also future, only for training the model
        order_demand = order_demand[order_demand.index.year >= 2015]
        order_demand = order_demand[order_demand.index.year <= 2024]
        return order_demand
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return pd.Series()


# Train ARIMA model
def train_arima(data):
    try:
       model = ARIMA(data, order=(5,1,0))  # Example parameters, adjust as needed. Important to tune.
       model_fit = model.fit()
       return model_fit
    except Exception as e:
        print(f"Error during ARIMA training: {e}")
        return None


# Predict demand
def predict_demand(model_fit, steps):
    try:
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


# Function to get yearly historical data for a product
def get_yearly_demand(product_code):
    product_history = data[data['Product_Code'] == product_code].copy()

    if product_history.empty:
        return None

    try:
        product_history['Date'] = pd.to_datetime(product_history['Date'], errors='coerce')
        product_history.set_index('Date', inplace=True)
        yearly_demand = product_history['Order_Demand'].resample('Y').sum()
        #Filter historical demand year also
        yearly_demand = yearly_demand[yearly_demand.index.year >= 2015]
        yearly_demand = yearly_demand[yearly_demand.index.year <= 2024]
        # yearly_demand.name = 'Yearly Demand'  # Set the series name for better display
        return yearly_demand

    except Exception as e:
        print(f"Error getting yearly demand: {e}")
        return None

# Function to generate historical plot
def create_historical_plot(historical_data, product_code):
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # Ensure data is not empty
    if not historical_data.empty:
        # Convert DatetimeIndex to integers (years)
        years = historical_data.index.year.astype(int)
        ax.plot(years, historical_data.values, label='Historical Demand', marker='o', linestyle='-', color='blue')
    else:
        ax.text(0.5, 0.5, 'No historical data available for the selected product between 2015-2024.', ha='center', va='center', fontsize=12)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Product Demand', fontsize=12)
    ax.set_title(f'Historical Yearly Demand for {product_code} ', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Set x-axis limits
    min_year = 2015
    max_year = 2024
    ax.set_xlim(min_year - 1, max_year + 1)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()

    image_png = buffer.getvalue()
    graphic = base64.b64encode(image_png).decode('utf-8')

    return graphic
# Function to generate forecast plot
def create_forecast_plot(forecast_data, product_code):
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    forecast_years = [int(year) for year in forecast_data.keys()]  # Ensure forecast years are integers
    forecast_years = [year for year in forecast_years if 2025 <= year <= 2029]  # Limit forecast years to 2025-2029

    forecast_values = {year: forecast_data[str(year)] for year in forecast_years}  # Creating new dict value with keys

    # Ensure data is not empty
    if forecast_values:
         ax.plot(forecast_years, forecast_values.values(), label='Forecasted Demand ', marker='x', linestyle='--', color='red')

        # Annotate the forecast values
         for year, demand in forecast_values.items():
            ax.annotate(f'{demand:.0f}',  # Format the demand
                        xy=(int(year), demand),  # Coordinates of the annotation
                        xytext=(0, 5),  # Offset from the point (in points)
                        textcoords="offset points",
                        ha='center', va='bottom',  # Alignment
                        fontsize=8, color='green')
    else:
        ax.text(0.5, 0.5, 'No forecast data available between 2025-2029.', ha='center', va='center', fontsize=12)


    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Product Demand', fontsize=12)
    ax.set_title(f'Forecasted Yearly Demand for {product_code} ', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    # Set x-axis limits
    min_year = 2025
    max_year = 2029
    ax.set_xlim(min_year - 1, max_year + 1)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()

    image_png = buffer.getvalue()
    graphic = base64.b64encode(image_png).decode('utf-8')

    return graphic

# Function to get the demand for printing
def get_printable_demand(product_code):
    return None #Return None to remove the table

# HTML template for product selection and history
template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Product Demand Prediction</title>
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            font-family: Arial, sans-serif;
            /* Fallback color */
            background-color: #f0f8ff;

            /* Create the linear gradient. */
            background-image: url('https://www.marketing91.com/wp-content/uploads/2018/03/What-is-Market-Demand.jpg');

            /* Set the size and behavior of the image */
            background-size: cover;

            /* Animation properties */
            animation: background-pan 10s linear infinite;
            color: #333;
        }

        @keyframes background-pan {
            from {
                background-position: 0% 0%;
            }
            to {
                background-position: 100% 100%;
            }
        }
        h1 { color: navy; }
        form { margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; }
        select { width: 200px; padding: 8px; margin-bottom: 10px; }
        input[type="submit"] { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        input[type="submit"]:hover { background-color: #3e8e41; }
        .forecast { margin-top: 20px; border: 1px solid #ddd; padding: 10px; background-color: rgba(255, 255, 255, 0.8); color: black; }
        .error { color: red; }
        ul { list-style-type: none; padding: 0; }
        li { margin-bottom: 5px; }
        .table-container { overflow-x: auto; } /* Enable horizontal scrolling for tables */
        @media print {
            body * {
                visibility: visible !important;
            }
            #section-to-print, #section-to-print * {
                visibility: visible !important;
            }
            #printButton {
                display: none !important;
            }
            body {
                -webkit-print-color-adjust: exact !important;
            }
        }
        img {
            max-width: 100%; /* Make sure images don't overflow their container */
            height: auto;
            display: block; /* Remove extra space below image */
            margin: 0 auto; /* Center the image */
        }
        .plot-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 20px;
        }
        .plot-container h2 {
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1><b>Product Demand Prediction</b></h1>
        <form action="/" method="post">
            <div class="form-group">
                <label for="product_code">Select Product:</label>
                <select id="product_code" name="product_code" class="form-control">
                    {% for product in products %}
                    <option value="{{ product }}">{{ product }}</option>
                    {% endfor %}
                </select>
            </div>
            <button type="submit" class="btn btn-primary">Predict</button>
        </form>

        {% if forecast %}
        <div class="forecast">
            <h2>Prediction for {{ product_code }}:</h2>
            <ul>
            {% for year, demand in forecast.items() %}
                <li><b>{{ year }}:</b> {{ demand|round(2) }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}

        {% if error %}
        <p class="error">{{ error }}</p>
        {% endif %}

        {% if historical_plot_url %}
        <div class="plot-container">
            <h2>Historical Demand for {{ product_code }}:</h2>
            <img src="data:image/png;base64,{{ historical_plot_url }}" alt="Historical Demand">
        </div>
        {% endif %}

        {% if forecast_plot_url %}
        <div class="plot-container">
            <h2>Predicted Demand for {{ product_code }} :</h2>
            <img src="data:image/png;base64,{{ forecast_plot_url }}" alt="Forecasted Demand">
        </div>
        {% endif %}

    <button id="printButton" class="btn btn-info" onclick="window.print()">Print Page</button>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.3/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
'''


# Get unique product codes
products = data['Product_Code'].unique().tolist()


@app.route('/', methods=['GET', 'POST'])
def index():
    forecast = None
    error = None
    yearly_demand_table = None
    historical_plot_url = None
    forecast_plot_url = None # Initialize forecast_plot_url
    printable_demand = None #Initialize print demand

    if request.method == 'POST':
        product_code = request.form['product_code']
        series = prepare_data(product_code)

        #Get yearly demand
        historical_data = get_yearly_demand(product_code)

        if series.empty or historical_data is None or historical_data.empty:
            error = "No data found for this product or no data between 2015-2024."
            return render_template_string(template, products=products, error=error, yearly_demand_table=yearly_demand_table, product_code=product_code, historical_plot_url=historical_plot_url, forecast_plot_url=forecast_plot_url, historical_demand_table = printable_demand)

        model_fit = train_arima(series)

        if model_fit is None:
            error = "Failed to train ARIMA model."
            return render_template_string(template, products=products, error=error, yearly_demand_table=yearly_demand_table, product_code=product_code, historical_plot_url=historical_plot_url, forecast_plot_url=forecast_plot_url, historical_demand_table = printable_demand)

        forecast_values = predict_demand(model_fit, steps=5)  # Predict for next 5 years


        if forecast_values is None:
            error = "Failed to generate forecast."
            return render_template_string(template, products=products, error=error, yearly_demand_table=yearly_demand_table, product_code=product_code, historical_plot_url=historical_plot_url, forecast_plot_url=forecast_plot_url, historical_demand_table = printable_demand)

        # Convert forecast to a dictionary with date labels for the template
        forecast = {str(series.index[-1].year + i + 1): demand for i, demand in enumerate(forecast_values)}
        # Filter forecast to 2025-2029
        forecast = {year: demand for year, demand in forecast.items() if 2025 <= int(year) <= 2029}
        historical_plot_url = create_historical_plot(historical_data, product_code)
        forecast_plot_url = create_forecast_plot(forecast, product_code)
        printable_demand = get_printable_demand(product_code)


        return render_template_string(template, products=products, product_code=product_code, forecast=forecast, yearly_demand_table=yearly_demand_table, historical_plot_url=historical_plot_url, forecast_plot_url=forecast_plot_url, historical_demand_table = printable_demand)

    return render_template_string(template, products=products, forecast=forecast, error=error, yearly_demand_table=yearly_demand_table, historical_plot_url=historical_plot_url, forecast_plot_url=forecast_plot_url,  historical_demand_table = printable_demand)


if __name__ == "__main__":
    app.run(debug=True)
