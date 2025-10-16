try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

import pandas as pd


def load_Student_Mark() -> pd.DataFrame:
    """Return the dataset Student Mark.
      The data consists of Marks of students including their study time & number of courses. 
      The dataset is downloaded from UCI Machine Learning Repository.

    Number of Instances: 100
    Number of Attributes: 3 including the target variable.

    """
    # Use importlib.resources for modern Python versions
    dataset_file = files('desReg.dataset').joinpath('Student_Marks.csv')
    return pd.read_csv(dataset_file)