# Symbolic Regression (SR) for Knowledge Distillation (KD)

Using symbolic regression with [pySR](https://github.com/MilesCranmer/PySR) as a knowledge distillation tool in order to compress a neural netweork to deploy it on FPGA.

Run `sr.py` to perform the symbolic regression, internally uses PCA to go from 57-features input to 10.
Or run `sr_reduced.py` to perform the symbolic regression on 9-feature dataset reduced "by hand".
