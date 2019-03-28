from modelling import Modelling
import pandas as pd

# Set file to process, read it in, and create a Model for it
file_name = "docs/description.csv"
description = pd.read_csv(file_name)
model = Modelling(description)