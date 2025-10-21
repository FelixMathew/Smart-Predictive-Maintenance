import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

# --- Page Configuration ---
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide"
)

# --- Spark Session and Model Loading (Cached to run only once) ---
@st.cache_resource
def load_spark_model():
    """
    Initializes a Spark session with an ultra-lightweight configuration for deployment.
    """
    # --- THIS IS THE FINAL, MOST AGGRESSIVE FIX ---
    # We are forcing Spark to use only one core and the absolute minimum memory.
    spark = SparkSession.builder \
        .appName("PredictiveMaintenanceWebApp") \
        .master("local[1]") \
        .config("spark.driver.memory", "768m") \
        .config("spark.driver.memoryOverhead", "256m") \
        .config("spark.executor.memory", "768m") \
        .config("spark.sql.shuffle.partitions", "1") \
        .getOrCreate()
    
    model_path = "model/pyspark_rf_model"
    model = RandomForestClassificationModel.load(model_path)
    return spark, model

# Load the resources
try:
    spark, model = load_spark_model()
except Exception as e:
    st.error(f"Error loading Spark model: {e}")
    st.info("This can happen during the first startup. The app will likely succeed on the next automatic retry.")
    st.stop()


# --- App Title and Description ---
st.title("🔧 Smart Predictive Maintenance Dashboard (PySpark)")
st.markdown("Predict **machine failure** by providing real-time sensor data. This app uses a RandomForest model trained with PySpark.")

# --- Sidebar for User Input ---
st.sidebar.header("Input Sensor Data")

def user_inputs():
    """Creates sliders in the sidebar for all features the model was trained on."""
    air_temp = st.sidebar.slider('Air temperature [K]', 295.0, 305.0, 300.1, 0.1)
    process_temp = st.sidebar.slider('Process temperature [K]', 305.0, 315.0, 310.2, 0.1)
    rpm = st.sidebar.slider('Rotational Speed [rpm]', 1100, 3000, 1500)
    torque = st.sidebar.slider('Torque [Nm]', 0.0, 80.0, 40.5, 0.1)
    tool_wear = st.sidebar.slider('Tool Wear [min]', 0, 260, 55)
    
    data = {
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rpm,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }
    return data

input_data = user_inputs()

# --- Prediction Logic ---
if st.sidebar.button("Predict Failure"):
    # 1. Create a PySpark DataFrame from the user's input
    schema = StructType([
        StructField("Air temperature [K]", DoubleType(), True),
        StructField("Process temperature [K]", DoubleType(), True),
        StructField("Rotational speed [rpm]", IntegerType(), True),
        StructField("Torque [Nm]", DoubleType(), True),
        StructField("Tool wear [min]", IntegerType(), True),
    ])
    new_df = spark.createDataFrame([list(input_data.values())], schema)

    # 2. Assemble features
    features_list = list(input_data.keys())
    assembler = VectorAssembler(inputCols=features_list, outputCol="features")
    prepared_df = assembler.transform(new_df)

    # 3. Make prediction
    prediction = model.transform(prepared_df)
    result = prediction.select("prediction", "probability").collect()[0]
    failure_prediction = result['prediction']
    confidence_score = result['probability'][int(failure_prediction)]

    # 4. Display the result
    st.subheader("🧠 Prediction Result")
    if failure_prediction == 1.0:
        st.error(f"🚨 High Risk: Machine Failure Predicted! (Confidence: {confidence_score:.2%})", icon="🚨")
    else:
        st.success(f"✅ Low Risk: No Failure Predicted. (Confidence: {confidence_score:.2%})", icon="✅")

