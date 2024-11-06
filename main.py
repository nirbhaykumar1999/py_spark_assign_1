from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    DoubleType,
)
from pyspark.sql.functions import col, mean, format_number, avg, count

file_name = "insurance_sample.csv"

# Create a Spark session
spark = SparkSession.Builder().appName("Spart_Assignment_1").getOrCreate()

# Read the CSV file into a DataFrame
"""
Q2: The csv file is with headers. How can we set the column names using the headers in the file?
Ans: By passing header=true
"""
df = spark.read.csv(file_name, header=True, inferSchema=True)
# Show the first few rows of the DataFrame
df.show(10)

"""
Q3.1 What type(s) do the columns have?

|-- policyID: string (nullable = true)
|-- statecode: string (nullable = true)
|-- county: string (nullable = true)
|-- eq_site_limit: string (nullable = true)
|-- hu_site_limit: string (nullable = true)
|-- fl_site_limit: string (nullable = true)
|-- fr_site_limit: string (nullable = true)
|-- tiv_2011: string (nullable = true)
|-- tiv_2012: string (nullable = true)
|-- line: string (nullable = true)
|-- construction: string (nullable = true)
|-- point_granularity: string (nullable = true)

"""
df.printSchema()

"""
Q3.2 How can we make spark determine the schema more accurately?

To improve the accuracy of schema inference in Spark:
1. Use inferSchema=True to allow Spark to automatically detect the data types.
2. Adjust samplingRatio for better inference in large datasets.
3. Define an explicit schema using StructType for the most accurate schema.
"""


"""
Q3.3 How can we define the schema explicitly?
"""

data_schema = StructType(
    [
        StructField("policyId", IntegerType(), True),
        StructField("statecode", StringType(), True),
        StructField("county", StringType(), True),
        StructField("eq_site_limit", DoubleType(), True),
        StructField("hu_site_limit", DoubleType(), True),
        StructField("fl_site_limit", DoubleType(), True),
        StructField("fr_site_limit", DoubleType(), True),
        StructField("tiv_2011", DoubleType(), True),
        StructField("tiv_2012", DoubleType(), True),
        StructField("line", StringType(), True),
        StructField("construction", StringType(), True),
        StructField("point_granularity", IntegerType(), True),
    ]
)

df = spark.read.csv(file_name, header=True, schema=data_schema)
df.show(5)
df.printSchema()


# Q4 Pritn all the distinct state codes
df.select("statecode").distinct().show()

# Q5 Find the policy id with highest eq_site_limit value
df.sort(col("eq_site_limit").desc()).limit(1).select("policyID").show()

"""
Q6
Show policy ID and a custom column with value that is combination of 2 columns 
ie tiv_2012 * point_granularity .Name of this column should be “HighTv”. 
Order in increasing order and show the first 20 rows of the dataframe.
"""
df.withColumn("HighTv", df["tiv_2012"] * df["point_granularity"]).select(
    "policyID", "HighTv"
).orderBy(col("HighTv").asc()).limit(20).show()

"""
Using raw query

df.selectExpr("policyID", "(tiv_2012 * point_granularity) AS HighTv") \
  .orderBy("HighTv").limit(20).show()
"""

"""
Q7
Calculate the mean value of column fl_site_limit and fill all the null values in 
column fl_site_limit with the mean value calculated in the previous step.
"""
mean_value = df.agg(mean("fl_site_limit")).alias("Mean Value").collect()[0][0]
print(f"Mean Value: {mean_value}")

df = df.fillna({"fl_site_limit": mean_value})
df.show(5)

"""
Q8
In the results of the previous question select only PolicyId and fl_site_limit columns. 
Update the values for fl_site_limit column to be formatted to only 2 decimal places.
"""
df = df.withColumn("fl_site_limit", format_number(df["fl_site_limit"], 2))
df.show(5)


"""
Q9
Find average hu_site_limit for each county
"""

df.groupBy("county").agg(avg("hu_site_limit").alias("avg_hu_site_limit")).select(
    "county", "avg_hu_site_limit"
).show(10)

"""
Q10
Count the number of rows for each construction type
"""

construction_counts = df.groupBy("construction").count()
construction_counts.show(10)


"""
Q11
Filter the above results for all constructions which have count > 10k.
"""
construction_counts.filter("count > 10000").show()

"""
Q12
Create a temporary view with this dataframe 
and calculate the number of rows for each construction type using SQL query.
"""
df.createOrReplaceTempView("temp_view")

construction_counts_using_raw = spark.sql(
    """
    select construction, count(*) as count from temp_view group by construction
"""
)
construction_counts_using_raw.show()

"""
Q13
Create an inner join of the dataframe in the below code and the one we have been using above and show the joined dataframe.
construction_df = spark.createDataFrame([
  {"construction": "Reinforced Concrete", "allowed": True},
  {"construction": "Wood", "allowed": False},
  {"construction": "Steel Frame", "allowed": True},
  {"construction": "Masonry", "allowed": False},
  {"construction": "Reinforced Masonry", "allowed": True}
])
construction_df.show()

"""

construction_df = spark.createDataFrame(
    [
        {"construction": "Reinforced Concrete", "allowed": True},
        {"construction": "Wood", "allowed": False},
        {"construction": "Steel Frame", "allowed": True},
        {"construction": "Masonry", "allowed": False},
        {"construction": "Reinforced Masonry", "allowed": True},
    ]
)

# Simplified join using `on` parameter
construction_counts_using_raw.join(construction_df, on="construction", how="inner").show()

spark.stop()
