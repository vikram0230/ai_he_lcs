import argparse
import pandas as pd
import joblib
import time

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', 
        help="CSV file containing the data for training.")
    parser.add_argument('truth',
        help="The column name in the data set for the ground truth")
    parser.add_argument('-o', '--outfile', default='model.joblib',
        help="Name of the joblib file to store the trained model.",
        required=False)
    parser.add_argument('-f', '--features', nargs='+',
        help="Optional list of features to use for training. " + 
        "If not specified, all columns other than the ground truth " +
        "will be used as features for training.",
        required=False,
        default=None)
    args = parser.parse_args()
    return args

def main():
    args = get_cli_args()
    df = pd.read_csv(args.data)

    if args.features is None:
        X = df[df.columns.difference([args.truth]) + [args.truth]]
    else:
        X = df[args.features + [args.truth]]
    
    X = X.dropna(subset=[args.truth])
    y = X[args.truth]
    X = X.drop(columns=[args.truth])

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', C=1, class_weight='balanced', probability=True))
    ])
    model.fit(X, y)
    joblib.dump(model, args.outfile)

start = time.time()
main()
end = time.time()
print(f"Completed execution in {end-start:.2f} seconds.")