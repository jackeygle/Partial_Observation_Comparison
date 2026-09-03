"""数据管线:ATC 的原始 CSV -> 每天一个 h5 -> 4 通道网格。

见 DOC_data_pipeline.md。三个方法都读这一步的产物，谁都不该自己再解析 CSV。
"""
