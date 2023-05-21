import shutil
import os

shutil.unpack_archive("numerical_results.zip")
os.remove("numerical_results.zip")
