# Documentation: Filtering NLST data, then generating ROC curves based on Sybil predictions, and calculting AUC

*Last updated 11/27/2023 by Abdul Zakkar*

Find the Python script `sybil_nlst_filter_to_roc.py` here. (link to be added)

### Script Arguments

- Sybil predictions- A CSV file with the format described here. (link to be added)
- Truth table- A CSV file with the format described here. (link to be added)
	- This table includes information regarding whether a patient had a diagnosis of lung cancer by year n. Sybil provides probabilities meant to predict this boolean (True/False) value.
- Method for choosing CT scan kernel (i.e. convolution filter), options:
	- Average the probability prediction across all kernels.
	- Choose kernel with highest prediction probability.
	- Choose kernel with lowest prediction probability.
	- Choose kernel based on a provided priority list.
- Filters, formatted as such:
	- `filter_name:value:operator`
	- Examples: `gender:1:=`, `race:2:!=`, `age:65:>=`

### Outputs

- A PNG of the ROC curve, labeled based on filters, and including the AUC calculation.
- A CSV output with various confusion matrices, as well as the Area Under Curve (AUC) value.
	- Metadata on the filters will also be included.
	- The confusion matrices will use various Sybil probability cutoffs.
	- By default, these cutoffs will range from a probability of 0.1 to 0.9, with steps of 0.1.
- Below is an example representation of the CSV file.

| Filters: |   |   |
|---|---|---|
| Race | 2 | Equal to |
| Gender | 2 | Equal to |
| Age | 65 | Greater than or equal to |
|   |   |   |
| Probability cutoff: | 0.1 |   |
|   | Actual positive | Actual negative |
| Predicted Positive | TP | FP |
| Predicted Negative | FN | TN |
| Probability cutoff: | 0.2 |   |
|   | Actual positive | Actual negative |
| Predicted Positive | TP | FP |
| Predicted Negative | FN | TN |
| ... | ... | ... |
| Area Under Curve | 0.8888 (example value) |   |

[^1]



[^1]: TP = True Positive, FP = False Positive, FN = False Negative, TN = True Negative.
