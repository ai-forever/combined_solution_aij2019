import os
import shutil
from urllib.request import urlopen
from zipfile import ZipFile


def download_aij_data(data_url, data_zipfile):
    to_save = os.getcwd()
    print("Downloading {} from {}".format(data_zipfile, data_url))
    with urlopen(data_url) as response, open(data_zipfile, "wb") as output:
        shutil.copyfileobj(response, output)
        print("Extracting data...")
        with ZipFile(data_zipfile, "r") as to_extract:
            to_extract.extractall(to_save)
    print("Done")


def main():
    data_url = "http://bit.ly/2ql5FYC"
    data_zipfile = "agi_data.zip"
    download_aij_data(data_url, data_zipfile)


if __name__ == "__main__":
    main()
