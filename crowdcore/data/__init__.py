"""Data pipeline: ATC's raw CSVs -> one h5 per day -> a 4-channel grid.

See DOC_data_pipeline.md. All three methods read this step's output; none should
parse the CSVs itself.
"""
