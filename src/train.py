import pandas as pd
import hydra
from omegaconf import DictConfig
import utils  # relative import
from sklearn.preprocessing import LabelEncoder

@hydra.main(version_base=None, config_path="../confs", config_name="preprocessing")
def main(cfg: DictConfig):
    # Get project root
    project_root = utils.get_original_cwd()

    # Build full path to CSV
    csv_path = f"{project_root}/{cfg.dataset.data}"
    print(csv_path)

    # Read data
    df = pd.read_csv(csv_path, encoding=cfg.dataset.encoding)
    print(f"Loaded dataset with shape: {df.shape}")
    print(df.columns)

    # Drop unwanted features
    drop_cols = cfg.variables.drop_features
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Clean numeric columns (remove commas, convert to float)
    numeric_cols = cfg.variables.numerical_vars_from_numerical
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Encode categorical columns
    # 1️⃣ Label encoding
    for col in cfg.variables.categorical_label_extraction:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # 2️⃣ One-hot encoding
    onehot_cols = [col for col in cfg.variables.categorical_onehot if col in df.columns]
    df = pd.get_dummies(df, columns=onehot_cols)

    # Separate features and target
    target_col = cfg.target if isinstance(cfg.target, str) else cfg.target.target
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe columns.")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train Decision Tree if selected
    if cfg.pipeline.pipeline01.lower() == "decisiontree":
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        print("Training complete!")
    else:
        print(f"Pipeline {cfg.pipeline.pipeline01} not implemented.")

if __name__ == "__main__":
    main()
