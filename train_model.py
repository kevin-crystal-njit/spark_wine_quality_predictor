from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import when

# Create a Spark session
spark = SparkSession.builder.appName("MLProject").getOrCreate()

# Load the dataset from S3
data = spark.read.csv("s3://kevin-project-2-files/TrainingDataset.csv", header=True, inferSchema=True, sep=";")

# Clean column names
data = data.toDF(*[col.strip('"') for col in data.columns])

# Assemble feature vector (use ALL columns except "quality")
feature_cols = [col for col in data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembled_data = assembler.transform(data)

# Logistic Regression
binary_data = assembled_data.withColumn("binaryLabel", when(data["quality"] > 5, 1).otherwise(0))

logr = LogisticRegression(featuresCol="features", labelCol="binaryLabel")
logr_model = logr.fit(binary_data)

print("\n=== Logistic Regression Coefficients ===")
print(f"Coefficient: {logr_model.coefficients}")
print(f"Intercept: {logr_model.intercept}")

# Save Logistic Regression model to S3
logr_model.write().overwrite().save("s3://kevin-project-2-files/logistic_model")

# Stop Spark session
spark.stop()
