from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Initialize Spark Session with the Hadoop path directly configured
spark = SparkSession.builder \
    .appName("SmartPredictiveMaintenance") \
    .config("spark.driver.extraJavaOptions", "-Dhadoop.home.dir=C:/Hadoop") \
    .getOrCreate()

print("✅ Spark session created.")

# 2. Load the dataset using the correct relative path
try:
    data = spark.read.csv("data/ai4i2020.csv", header=True, inferSchema=True)
    print("✅ Dataset loaded successfully.")
    print("Schema of the loaded data:")
    data.printSchema()

    # 3. Define features using the EXACT column names from the CSV
    features_list = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]

    # 4. Assemble features into a single vector column
    assembler = VectorAssembler(inputCols=features_list, outputCol="features")
    data_with_features = assembler.transform(data)

    # 5. Rename the target column to "label"
    data_final = data_with_features.withColumnRenamed("Machine failure", "label")

    # 6. Split data into training and testing sets
    train_data, test_data = data_final.randomSplit([0.8, 0.2], seed=42)
    print(f"\nData split into {train_data.count()} training rows and {test_data.count()} test rows.")

    # 7. Define and train the RandomForest model
    print("⏳ Training the RandomForest model...")
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100)
    model = rf.fit(train_data)
    print("✅ Model training complete.")

    # 8. Make predictions on the test data
    predictions = model.transform(test_data)

    # 9. Evaluate the model's accuracy
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("------------------------------------------")
    print(f"🎯 Model Accuracy: {accuracy*100:.2f}%")
    print("------------------------------------------")


    # 10. Save the trained model to the 'model' folder
    model_path = "model/pyspark_rf_model"
    model.write().overwrite().save(model_path)
    print(f"✅ Model saved to: {model_path}")

except Exception as e:
    print(f"❌ An error occurred: {e}")

finally:
    # 11. Stop the Spark Session
    spark.stop()
    print("🛑 Spark session stopped.")