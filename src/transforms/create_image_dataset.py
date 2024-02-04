from pyspark.sql import SparkSession, DataFrame, functions as F, types as T, Window

# Run from the src/ folder
NORMAL_FACES_IMAGES_PATH = "../data/normal_faces/lfw-deepfunneled/lfw-deepfunneled/"

FAKE_FACES_IMAGES_PATH = "../data/fake_faces"

NORMAL_NAMES_PATH = "../data/normal_faces/lfw_allnames.csv"

OUTPUT_DATASET_PATH = "../data/testing_dataset/"

DF_SIZE = 100

IMAGE_COLUMNS = [
    "image.origin",
    "image.height",
    "image.width",
    "image.nChannels",
    "image.data",
]

spark = SparkSession.builder.master('local[*]').config("spark.driver.memory", "5g").getOrCreate()



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
    ).withColumnRenamed("data", "image_data").drop("origin")


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

    w = Window.partitionBy("person_name")
    df = (
        image_df.select(
            *IMAGE_COLUMNS,
            F.lit(False).alias("is_deepfake"),
        )
        .withColumnRenamed("data", "image_data")
        .withColumn("person_name", F.element_at(F.split("origin", "/"), -2))
        .withColumn("__random", F.rand(seed=43))
        .orderBy("__random")
        # For our current purposes, limit to 100 rows
        .limit(DF_SIZE)
        .drop("__random", "origin")
        .withColumn(
            "image_count_for_person",
            F.count("person_name").over(w)
        )
    )

    return df


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
