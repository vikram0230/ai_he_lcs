import argparse
import pandas as pd

from models import plcom2012
# from Evaluation import generate_figure

MODEL_DICT = {
    'plcom2012': plcom2012
}

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", 
        choices=[
            'plcom2012',
            'svm6',
            'svm11',
            'sybil'
        ],
        help="Name of the ML model to test.")
    parser.add_argument("testset",
        help="CSV file containing the test set.")
    parser.add_argument('-s', "--submodel",
        help="Name of the submodel to test.",
        default=None)
    parser.add_argument('-o', "--outdir",
        help="The directory in which to generate the figure.")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_cli_args()

    # Get model
    model = MODEL_DICT[args.model]
    if type(model) is dict:
        if args.submodel == None:
            print(
                "The specified ML model must have a chosen submodel." +
                f"\nChoices: {list(model.keys())}"
            )
            exit(1)
        model = model[args.submodel]

    # Get test set
    test = pd.read_csv(args.testset)

    # Generate figure
    generate_figure(model, test, 
        output_directory=args.outdir
    )