## Setting up the data

The benchmarking scripts expects 2 files to be present in `data/demand`.

`data/demand/train.csv` : training data

`data/demand/test_full.csv` : testing data

To setup the data for benchmarking:

1. Use the `generate_data.py` script to generate synthetic data for `demand/train.csv` and `demand/test_full.csv` after activating a conda environment. Stock environment is used as reference below.

```shell
conda activate demand_stock
python generate_data.py
```