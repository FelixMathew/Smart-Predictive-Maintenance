from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

# 1. Initialize Spark Session with Hadoop configuration
spark = SparkSession.builder \
    .appName("PredictiveMaintenancePrediction") \
    .config("spark.driver.extraJavaOptions", "-Dhadoop.home.dir=C:/Hadoop") \
    .getOrCreate()

# 2. Load the saved model
model_path = "model/pyspark_rf_model"
try:
    loaded_model = RandomForestClassificationModel.load(model_path)
    print(f"✅ Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    spark.stop()
    exit()

# 3. Create new, unseen data to make predictions on
# This simulates new sensor readings from a machine.
new_data_schema = StructType([
    StructField("Air temperature [K]", DoubleType(), True),
    StructField("Process temperature [K]", DoubleType(), True),
    StructField("Rotational speed [rpm]", IntegerType(), True),
    StructField("Torque [Nm]", DoubleType(), True),
    StructField("Tool wear [min]", IntegerType(), True),
])

new_data = [
    # A machine operating under high stress - likely to fail
    (308.1, 312.5, 1250, 75.3, 210),
    # A machine operating normally
    (299.5, 309.8, 1550, 38.1, 25),
]

new_df = spark.createDataFrame(data=new_data, schema=new_data_schema)
print("\nNew data to predict:")
new_df.show()

# 4. Prepare the new data using the same VectorAssembler steps
# The input column names MUST be the same as during training.
features_list = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]
assembler = VectorAssembler(inputCols=features_list, outputCol="features")
prepared_df = assembler.transform(new_df)

# 5. Use the loaded model to make predictions
predictions = loaded_model.transform(prepared_df)

# 6. Show the results
print("\nPredictions:")
# The 'prediction' column shows the model's output (0 = No Failure, 1 = Failure)
predictions.select(
    "features",
    "prediction"
).show()

# Stop the Spark Session
spark.stop()
print("🛑 Spark session stopped.")