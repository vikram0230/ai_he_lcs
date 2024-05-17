import argparse
import pandas as pd

from models import Models
from evaluate import generate_figure

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("modeldir",
        help="Directory containing models.")
    parser.add_argument("model", 
        help="Name of the ML model to test.")
    parser.add_argument("testset",
        help="CSV file containing the test set.")
    parser.add_argument('truth',
        help="The column name in the data set for the ground truth")
    parser.add_argument('-o', "--outdir",
        help="The directory in which to generate the figure.")
    parser.add_argument('-f', '--features', nargs='+',
        help="Optional list of features to use for training. " + 
        "If not specified, all columns other than the ground truth " +
        "will be used as features for training.",
        required=False,
        default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_cli_args()

    models = Models(args.modeldir)

    # Get test set
    df = pd.read_csv(args.testset)
    if args.features is None:
        X = df[df.columns.difference([args.truth]) + [args.truth]]
    else:
        X = df[args.features + [args.truth]]
    
    X = X.dropna(subset=[args.truth])
    y = X[args.truth]
    X = X.drop(columns=[args.truth])

    predictors = models.get_models(args.model)
    print(predictors)

    # Generate figure
    generate_figure(predictors, X, y, output=args.outdir)