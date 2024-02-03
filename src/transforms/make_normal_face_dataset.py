from pyspark.sql import SparkSession, DataFrame, functions as F
from enum import Enum

# Run from the src/ folder
NORMAL_FACES_IMAGES_PATH = "../data/normal_faces/lfw-deepfunneled/lfw-deepfunneled/"

NAMES_PATH = "../data/normal_faces/lfw_allnames.csv"

OUTPUT_DATASET_PATH = "../data/normal_faces/datasets/normal_faces"

DF_SIZE = 100

spark = SparkSession.builder.getOrCreate()


def main():
    image_df: DataFrame = spark.read.format("image").option("dropInvalid", True).option("recursiveFileLookup", "true").load(NORMAL_FACES_IMAGES_PATH)
    names_df: DataFrame = spark.read.format("csv").option("header", True).load(NAMES_PATH)
    df = (
        image_df
        .select(
            "image.origin",
            "image.height",
            "image.width",
            "image.nChannels",
            F.col("image.data").alias("image_data"),
            F.lit(False).alias("is_deepfake"),
        )
        .withColumn(
            "person_name",
            F.element_at(F.split("origin", "/"), -2)
        )
        .join(
            names_df.select(F.col("name").alias("person_name"), F.col("images").alias("image_count_for_person")),
            on="person_name",
            how="left"
        )
        .withColumn(
            "image_id",
            F.monotonically_increasing_id()
        )
    )

    # For our current purposes, limit to 100 rows
    df = (
        df
        .withColumn(
            "__random",
            F.rand(seed=43)
        )
        .orderBy("__random")
        .limit(DF_SIZE)
    )

    print("Writing dataframe...")
    df.write.parquet(OUTPUT_DATASET_PATH, mode="overwrite")
    print("Done!")
    

if __name__ == "__main__":
    main()