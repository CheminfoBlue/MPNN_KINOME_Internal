import pandas as pd
import sys

def get_test_data(input_data): 
    df_kinome = pd.read_csv(input_data)
    df_kinome_test = df_kinome[df_kinome['Split'] == 'test']
    return df_kinome_test.to_csv(index=False)

if __name__ == '__main__':
    input_data = './data/kinome_data_multiclass_current.csv'
    test_data = get_test_data(input_data)
    sys.stdout.write(test_data)
