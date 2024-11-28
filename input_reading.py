import pandas as pd


class InputReader:
    def __init__(self, data, dtype="csv"):
        self.data = data
        self.dtype = dtype

    def reader_logic(self):
        match self.dtype:
            case "csv":
                df = pd.read_csv(self.data, index_col=False)
            case "xlsx":
                df = pd.read_excel(self.data, index_col=False)
            case "xls":
                df = pd.read_excel(self.data, index_col=False)
            case _:
                return -1

        return df
