#########################################################
######                preprocess                    #####
#########################################################
# python preprocess.py -i data/iris.csv -o data/iris_processed.csv
# python preprocess.py -i data/car.csv -o data/car_processed.csv
# python preprocess.py -i data/adult.csv -o data/adult_processed.csv


import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='preprocess dataset.')
parser.add_argument('-i', '--input', help='Path of input dataset')
parser.add_argument('-o', '--output', help='Path of output dataset')

args = parser.parse_args()
in_path = args.input
out_path = args.output

in_data = pd.read_csv(in_path)
for column in in_data.columns:
    if in_data[column].dtypes == 'object':
        in_data[column] = in_data[column].astype('category').cat.codes
    else:
        in_data[column] = (in_data[column] - in_data[column].mean())/in_data[column].std()

in_data.to_csv(out_path, index=False)