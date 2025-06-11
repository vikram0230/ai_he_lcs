import argparse
from os.path import isfile
import pandas as pd

from models import Models
from evaluate import generate_results, generate_results_ci

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as font_manager

font_path = 'C:\Windows\Fonts\segoeui.ttf'
prop = font_manager.FontProperties(fname=font_path)
prop.set_weight = 'medium'
rcParams['font.family'] = prop.get_name()
rcParams['font.weight'] = 'medium'

rcParams['figure.dpi'] = 300

colors = [
    '#4EBFD4',
    '#F55356',
    '#FF97E7',
    '#8C4FC1',
    '#FFC636',
    '#A7E28E',
]

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("modeldir",
        help="Directory containing models.")
    parser.add_argument("model", 
        help="Name of the ML model to test." \
        "Treated as a regular expression and used to search for models.")
    parser.add_argument("testset",
        help="CSV file containing the test set.")
    parser.add_argument('truth',
        help="The column name in the data set for the ground truth")
    parser.add_argument('-o', "--outdir",
        help="The directory in which to generate the figure.")
    parser.add_argument('-f', '--features', nargs='+',
        help="Optional list of features to use for testing. " + 
        "If not specified, all columns other than the ground truth " +
        "will be used as features for testing.",
        required=False,
        default=None)
    parser.add_argument('-e', '--ensemble', action='store_true',
        help="Optional argument to use multiple test sets. " +
        "Any test set will be used. " +
        "This means that a 95%% confidence interval will be generated across " +
        "all test sets. Uses 'feature' model.")
    parser.add_argument('-v', '--verbose', action='store_true',
        help="Optional argument to provide more information during execution.")
    args = parser.parse_args()
    return args

def get_csvs(path, verbose):
    output = []
    MAX_ITER = 10000
    index = 1
    while index < MAX_ITER:
        current_path = path.replace('#', str(index))
        if not isfile(current_path):
            index -= 1
            break
        if verbose:
            print(f'Found {current_path}.')
        output.append(pd.read_csv(current_path))
        index += 1
    if verbose:
        print(f'Found {index} csvs.')
    return output

if __name__ == '__main__':
    args = get_cli_args()

    models = Models(args.modeldir)
    predictors = models.get_models(args.model)

    f = plt.figure(figsize=(6.5,3), dpi=144)
    ax_pr = f.add_subplot(121)
    ax_pr.set_xlabel("Recall", labelpad=-10)
    ax_pr.set_ylabel("Precision", labelpad=-10)
    ax_pr.set_xlim(-0.05, 1.05)
    ax_pr.set_ylim(-0.05, 1.05)
    ax_pr.set_xticks([0,1])
    ax_pr.set_yticks([0,1])

    ax_roc = f.add_subplot(122)
    ax_roc.set_xlabel("False positive rate", labelpad=-10)
    ax_roc.set_ylabel("True positive rate", labelpad=-10)
    ax_roc.set_xlim(-0.05, 1.05)
    ax_roc.set_ylim(-0.05, 1.05)
    ax_roc.set_xticks([0,1])
    ax_roc.set_yticks([0,1])

    if args.ensemble:
        # Create XY data set for each model.
        # 3D data- X (specificity), Y (senstivity), N year
        # Data is 4D when you include the model.
        # Then generate the image with CIs.
        dfs = get_csvs(args.testset, args.verbose)

        results = pd.DataFrame()
        index = 0
        for feature in args.features:
            truth = args.truth
            if '#' in truth:
                truth = truth.replace('#', feature[-1])
            X_list, y = [], []
            for inner_index, df in enumerate(dfs):
                X = df[[feature, truth]]
                X = X.dropna(subset=[truth])
                if inner_index == 0:
                    y = X[truth]
                X = X.drop(columns=[truth])
                X_list.append(X)
            print(f"Model (None)")
            print(f"Truth column: {truth}")
            print(f"Feature columns: {feature}")
            result = generate_results_ci(predictors['feature'], X_list, y, ax_pr, ax_roc,
                plot_label=f'Year {truth[-1]}',
                plot_color=colors[index % len(colors)],
                draw_roc_diagonal= index==0,
                z_index=6-index,
                n_points=1000,
                verbose=args.verbose)
            results = pd.concat([results, result])
            index += 1
        filename = args.outdir + '\\' + 'feature-' + args.testset.split('\\')[-1].split('.')[0]
        filename = filename.replace('#', 'N_ci')
        f.savefig(filename + '.svg', format='svg', bbox_inches='tight')
        results.to_csv(filename + '.csv', index=False)
        exit(0)
    
    df = pd.read_csv(args.testset)

    if args.model == 'feature':
        results = pd.DataFrame()
        index = 0
        for feature in args.features:
            truth = args.truth
            if '#' in truth:
                truth = truth.replace('#', feature[-1])
            X = df[[feature, truth]]
            X = X.dropna(subset=[truth])
            y = X[truth]
            X = X.drop(columns=[truth])
            print(f"Model (None)")
            print(f"Truth column: {truth}")
            print(f"Feature columns: {feature}")
            result = generate_results(predictors['feature'], X, y, ax_pr, ax_roc,
                plot_label=f'Year {truth[-1]}',
                plot_color=colors[index % len(colors)],
                draw_roc_diagonal= index==0,
                z_index=6-index,
                verbose=args.verbose)
            results = pd.concat([results, result])
            index += 1
        filename = args.outdir + '\\' + 'feature-' + args.testset.split('\\')[-1].split('.')[0]
        f.savefig(filename + '.svg', format='svg', bbox_inches='tight')
        results.to_csv(filename + '.csv', index=False)
        exit(0)

    results = pd.DataFrame()
    index = 0
    model_name = 'model'
    for name, model in predictors.items():
        if index == 0:
            model_name = name.split('_')[0]
        print(f"Model {name}")
        truth = args.truth
        if '#' in truth:
            truth = truth.replace('#', name[-1])
        print(f"Truth column: {truth}")
        if args.features is None:
            X = df[df.columns.difference([truth]) + [truth]]
            print(f"Feature columns: {df.columns.difference([truth])}")
        else:
            features = args.features.copy()
            for i in range(len(features)):
                if '#' in features[i]:
                    features[i] = features[i].replace('#', name[-1])
            print(f"Feature columns: {features}")
            X = df[features + [truth]]
        X = X.dropna(subset=[truth])
        y = X[truth]
        X = X.drop(columns=[truth])
        result = generate_results(model, X, y, ax_pr, ax_roc,
            plot_label=f'Year {truth[-1]}',
            plot_color=colors[index % len(colors)],
            draw_roc_diagonal= index==0,
            z_index=6-index,
            verbose=args.verbose)
        results = pd.concat([results, result])
        index += 1
    filename = args.outdir + '\\' + model_name + '-' + args.testset.split('\\')[-1].split('.')[0]
    f.savefig(filename + '.svg', format='svg', bbox_inches='tight')
    results.to_csv(filename + '.csv', index=False)
    exit(0)