from hdmf_ai import ResultsTable  # NOTE: because hdmf_ai modifies the hdmf common namespace, it is important that hdmf_ai is imported before pynwb
from hdmf.common import HERD
import numpy as np
from pynwb import NWBHDF5IO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

filepath = "/Users/rly/Documents/NWB_Data/dandisets/000409/sub-CSHL047/sub-CSHL047_ses-b52182e7-39f6-4914-9717-136db589706e_behavior+ecephys+image.nwb"
io = NWBHDF5IO(filepath, "r")
nwbfile = io.read()

# the NWB Units table stores information about the sorted single units (putative neurons) after preprocessing
# and spike sorting. each row represents a single unit. this dataset includes many metadata fields (table columns) for
# each unit.
units = nwbfile.units

# the Units table can be most readiy viewed as a pandas DataFrame
units_df = units.to_dataframe()
print(units_df)
print(f"There are {len(units_df)} units in this dataset.")

# run a simple classifier on the units data to predict the location (brain area) of the unit based on
# the amplitude, firing rate, spike count, and presence ratio. there are several ways to label the brain
# area of a unit. here, we use the label using the coarsest atlas, the Cosmos atlas, which has a total
# of 12 annotation regions. in this dataset, there are 5 unique Cosmos locations.
cosmos_location = units_df["cosmos_location"].to_numpy()
enc = LabelEncoder()
y = np.uint(enc.fit_transform(cosmos_location))
unique_labels = enc.classes_
print(f"There are {len(unique_labels)} unique Cosmos locations in this dataset.")

# split the data into training and test sets
# TODO integrate with sklearn.model_selection.train_test_split
proportion_train = 0.7
n_train_samples = int(np.round(proportion_train * len(units_df)))
n_test_samples = len(units) - n_train_samples
# train = 0, validate = 1, test = 2
tvt = np.array([0] * n_train_samples + [2] * n_test_samples)
np.random.shuffle(tvt)

feature_names = [
    "median_amplitude",
    "standard_deviation_amplitude",
    "firing_rate",
    "spike_count",
    "presence_ratio",
]
X = units_df[feature_names]
X_train = X[tvt == 0]
y_train = y[tvt == 0]
X_test = X[tvt == 2]
y_test = y[tvt == 2]

# run logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
predictions_all = logreg.predict(X)
prediction_proba_all = logreg.predict_proba(X)
print(prediction_proba_all)
score = logreg.score(X_test, y_test)
print(f"The logistic regression achieved a score of {score} on the test set!")

results_table = ResultsTable(
    name="logistic_regression_results",
    description="Results of a simple logisitic regression on the units table",
    n_samples=len(units),
)
results_table.add_tvt_split(tvt)
results_table.add_true_label(cosmos_location)  # use the text labels which will become an EnumData with uint encoding
results_table.add_predicted_probability(prediction_proba_all)
results_table.add_predicted_class(predictions_all)
results_table.add_samples(data=np.arange(len(X)), description="all the samples", table=units)
# TODO address len(id) mismatch when adding as first column
# TODO address warnings about mismatch with predefined spec
# TODO demonstrate adding custom column
results_table.add_column(name="custom_metadata", data=np.random.rand(len(X)), description="random data")
print(results_table.to_dataframe())

# add the results table to the in-memory NWB file
nwbfile.add_analysis(results_table)

# store metadata about the model
# NOTE: not the actual model, just a placeholder for demonstration purposes
results_table.pre_trained_model = "Bloom v1.3"
# annotate the model with a DOI using HDMF HERD
herd = HERD()
herd.add_ref(
    file=nwbfile,
    container=results_table,
    attribute="pre_trained_model",
    key=results_table.pre_trained_model,
    entity_id='doi:10.57967/hf/0003',
    entity_uri='https://doi.org/10.57967/hf/0003'
)
herd.to_zip(path='./HERD.zip')

# remove all the voltage recording raw data which is not needed for this analysis and takes up a lot of space
for x in list(nwbfile.acquisition.keys()):
    nwbfile.acquisition.pop(x)

with NWBHDF5IO("results.nwb", "w") as export_io:
    # TODO allow storage of results in a different file from input data but maintain DTR link
    export_io.export(src_io=io, nwbfile=nwbfile)

io.close()

with NWBHDF5IO("results.nwb", "r") as read_io:
    read_nwbfile = read_io.read()
    print(read_nwbfile.analysis["logistic_regression_results"].to_dataframe())
    # TODO check values
