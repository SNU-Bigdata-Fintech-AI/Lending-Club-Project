import pandas as pd

import feature_engineering as fe
import feature_selector as fs
import category_encoder as ce

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = fe.add_default_features(df)
    df = fe.clean_issue_date_format(df)
    df = fs.remove_missing_values(df)
    df = fs.remove_unnecessary_columns(df)
    df = fe.clean_term_format(df)
    df = fe.clean_emp_length(df)
    df = fe.clean_revol_util(df)
    df = fe.encode_sub_grade(df)
    df = ce.one_hot_encode_columns(df)
    df = fe.handle_low_missing_values(df)
    df = fe.handle_high_missing_values(df)
    df = fs.remove_inf_values(df)
    df = fe.log_transform_features(df)
    df = fe.binarize_count_features(df)

    return df
