from pyspark.sql import SparkSession, DataFrame
from src.constants import NORMAL_FACES_IMAGES_PATH

spark = SparkSession.builder.getOrCreate()

def main():
    df: DataFrame = spark.read.format("image").option("dropInvalid", True).option("recursiveFileLookup", "true").load(NORMAL_FACES_IMAGES_PATH)
    df = (
        df
        .select(
            "image.origin",
            "image.height",
            "image.width",
            "image.nChannels",
            "image.data",
        )
    )
    print(df.select("image.origin",
            "image.height",
            "image.width",
            "image.nChannels").head())
    print(df.count())

if __name__ == "__main__":
    main()