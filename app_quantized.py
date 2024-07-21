import os
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType
from scipy.optimize import newton
import datetime as dt
import time
from io import StringIO
import base64

os.environ['PYSPARK_PYTHON'] = 'C:\\Users\\Nilesh\\Desktop\\xirr-calculator\\.venv\\Scripts\\python.exe'


# Initialize the Spark session
# spark = SparkSession.builder \
#     .appName("XIRR Calculator") \
#     .master("local[*]") \
#     .config("spark.executor.memory", "4g") \
#     .config("spark.driver.memory", "4g") \
#     .getOrCreate()
spark = SparkSession.builder \
    .appName("XIRR Calculator") \
    .master("local[*]") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.default.parallelism", "4") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "1g") \
    .getOrCreate()


# Quantization parameters
AMOUNT_SCALE = 1000  # This will give us 3 decimal places of precision
DATE_ORIGIN = pd.Timestamp('1970-01-01')


def quantize_amount(amount):
    return int(amount * AMOUNT_SCALE)


def dequantize_amount(quantized_amount):
    return quantized_amount / AMOUNT_SCALE


def quantize_date(date):
    return (date - DATE_ORIGIN).days


def npv(rate, cash_flows):
    return sum(dequantize_amount(cf) / (1 + rate) ** (d / 365.0) for d, cf in cash_flows)


def calculate_xirr(cash_flows):
    try:
        return newton(lambda r: npv(r, cash_flows), x0=0.1, tol=1e-5, maxiter=1000)
    except RuntimeError:
        return np.nan


def xirr_pandas(df):
    df['QuantizedAmount'] = df['Amount'].apply(quantize_amount)
    df['QuantizedDate'] = df['Date'].apply(quantize_date)
    cash_flows = list(zip(df['QuantizedDate'], df['QuantizedAmount']))
    return calculate_xirr(cash_flows)


def xirr_spark(spark_df):
    num_cores = os.cpu_count()
    num_partitions = max(2, num_cores * 2)

    spark_df = spark_df.repartition(num_partitions)

    # Quantize Amount and Date
    spark_df = spark_df.withColumn('QuantizedAmount', (F.col('Amount') * F.lit(AMOUNT_SCALE)).cast(LongType()))
    spark_df = spark_df.withColumn('QuantizedDate', F.datediff(F.col('Date'), F.lit(DATE_ORIGIN.strftime('%Y-%m-%d'))))

    cash_flows = spark_df.select('QuantizedDate', 'QuantizedAmount').collect()
    cash_flows = [(row['QuantizedDate'], row['QuantizedAmount']) for row in cash_flows]

    return calculate_xirr(cash_flows)


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "XIRR Calculator"

# Layout of the app
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("XIRR Calculator"), className="text-center mt-3")),
    dbc.Row(dbc.Col(dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ))),
    dbc.Row(dbc.Col(html.Div(id='output-data-upload'))),
    dbc.Row(dbc.Col(html.Div(id='xirr-result', className='mt-3'))),
    dbc.Row(dbc.Col(html.Div(id='performance-comparison', className='mt-3'))),
])


# Callback to handle file upload and XIRR calculation
@app.callback(
    Output('output-data-upload', 'children'),
    Output('xirr-result', 'children'),
    Output('performance-comparison', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def update_output(contents, filename, date):
    if contents is None:
        return "", "", ""

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            df = pd.read_csv(StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(decoded)
        else:
            return "Unsupported file type", "", ""

        print("Loaded DataFrame:")
        print(df.head())
        print("DataFrame info:")
        print(df.info())

        # Ensure the DataFrame has the correct columns
        if 'Amount' not in df.columns or 'Date' not in df.columns:
            return "The file must contain 'Amount' and 'Date' columns", "", ""

        # Preprocess data
        df['Date'] = pd.to_datetime(df['Date'])
        df['Amount'] = df['Amount'].astype(float)

        # Calculate XIRR using pandas
        start_time = time.time()
        xirr_value_pandas = xirr_pandas(df)
        pandas_time = time.time() - start_time

        # Calculate XIRR using PySpark
        start_time = time.time()
        spark_df = spark.createDataFrame(df)
        xirr_value_spark = xirr_spark(spark_df)
        spark_time = time.time() - start_time

        performance_comparison = html.Div([
            html.P(f"Pandas calculation time: {pandas_time:.4f} seconds"),
            html.P(f"PySpark calculation time: {spark_time:.4f} seconds")
        ])

        return html.Div([
            html.H5(f"File: {filename}"),
            html.Hr(),
            dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True)
        ]), html.Div([
            html.P(f"Pandas XIRR: {xirr_value_pandas:.2%}"),
            html.P(f"PySpark XIRR: {xirr_value_spark:.2%}")
        ]), performance_comparison

    except Exception as e:
        print(f"Error in update_output: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return f"Error: {str(e)}", "", ""


if __name__ == '__main__':
    app.run_server(debug=True, port=8060)
