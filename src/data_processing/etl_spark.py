from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, expr, year, month
from pyspark.sql.window import Window
import yfinance

spark = SparkSession.builder.appName("Stock Analysis").getOrCreate()

# GET Stock DATA
symbol = 'PETR4.SA'
start_date = '2015-01-01'

data = yfinance.download('PETR4.SA', start=start_date)
data.columns = data.columns.droplevel(1)

# Converte para dataframe do Spark
data_spark_df = spark.createDataFrame(data.reset_index())

# Cria janelamentos de 7, 25 e 99 dias
window_7 = Window.orderBy("Date").rowsBetween(-6, 0)
window_25 = Window.orderBy("Date").rowsBetween(-24, 0)
window_99 = Window.orderBy("Date").rowsBetween(-98, 0)

# Adicionar colunas de médias móveis
data_spark_df = data_spark_df \
    .withColumn("m_avg_7", avg(col("Close")).over(window_7)) \
    .withColumn("m_avg_25", avg(col("Close")).over(window_25)) \
    .withColumn("m_avg_99", avg(col("Close")).over(window_99))

# Remover colunas indesejadas e valores nulos
data_spark_df = data_spark_df.drop("Dividends", "Stock Splits", "Volume").na.drop()

data_spark_df = data_spark_df.drop("Date")

# Salvar como parquet com particionamento por ano e mês
data_spark_df.write.mode("overwrite").parquet("data/petr4_10years.parquet")
