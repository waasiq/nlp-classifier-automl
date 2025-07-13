if __name__ == '__main__':
    # read the .npy file and print the array
    import argparse
    import numpy as np
    import csv

    with open('results/dataset=dbpedia/seed=42/test_preds.npy', 'rb') as f:
        test_preds = np.load(f)
    print(test_preds[:10])
