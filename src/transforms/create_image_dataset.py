from pyspark.sql import SparkSession, DataFrame, functions as F, types as T
from enum import Enum

# Run from the src/ folder
NORMAL_FACES_IMAGES_PATH = "../data/normal_faces/lfw-deepfunneled/lfw-deepfunneled/"

FAKE_FACES_IMAGES_PATH = "../data/fake_faces"

NORMAL_NAMES_PATH = "../data/normal_faces/lfw_allnames.csv"

OUTPUT_DATASET_PATH = "../data/normal_faces/datasets/normal_faces"

DF_SIZE = 100

IMAGE_COLUMNS = [
    "image.origin",
    "image.height",
    "image.width",
    "image.nChannels",
    "image.data",
]

spark = SparkSession.builder.getOrCreate()


def get_fake_faces_df() -> DataFrame:
    image_df: DataFrame = (
        spark.read.format("image")
        .option("dropInvalid", True)
        .option("recursiveFileLookup", "true")
        .load(FAKE_FACES_IMAGES_PATH)
    )

    return image_df.select(
        *IMAGE_COLUMNS,
        F.lit(True).alias("is_deepfake"),
        F.lit(1).alias("image_count_for_person"),
        F.lit(None).cast(T.StringType()).alias("person_name"),
    ).withColumnRenamed("data", "image_data")


def get_normal_faces_df() -> DataFrame:
    image_df: DataFrame = (
        spark.read.format("image")
        .option("dropInvalid", True)
        .option("recursiveFileLookup", "true")
        .load(NORMAL_FACES_IMAGES_PATH)
    )
    names_df: DataFrame = (
        spark.read.format("csv").option("header", True).load(NORMAL_NAMES_PATH)
    )
    df = (
        image_df.select(
            *IMAGE_COLUMNS,
            F.lit(False).alias("is_deepfake"),
        )
        .withColumnRenamed("data", "image_data")
        .withColumn("person_name", F.element_at(F.split("origin", "/"), -2))
        .join(
            names_df.select(
                F.col("name").alias("person_name"),
                F.col("images").alias("image_count_for_person"),
            ),
            on="person_name",
            how="left",
        )
    )

    # For our current purposes, limit to 100 rows
    return df.withColumn("__random", F.rand(seed=43)).orderBy("__random").limit(DF_SIZE)


def main():
    normal_faces = get_normal_faces_df()
    fake_faces = get_fake_faces_df()

    df = normal_faces.unionByName(fake_faces).withColumn(
        "image_id", F.monotonically_increasing_id()
    )

    print("Writing dataframe...")
    df.write.parquet(OUTPUT_DATASET_PATH, mode="overwrite")
    print("Done!")


if __name__ == "__main__":
    main()
