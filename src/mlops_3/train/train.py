import pandas as pd
import hydra
from omegaconf import DictConfig
from src import utils

@hydra.main(version_base=None, config_path="../configs", config_name="preprocessing")
def main(cfg: DictConfig):
    # Get project root
    project_root = utils.get_original_cwd()

    # Build full path to CSV
    csv_path = f"{project_root}/{cfg.dataset.data}"

    # Read data
    df = pd.read_csv(csv_path, encoding=cfg.dataset.encoding)
    print(f"Loaded dataset with shape: {df.shape}")

    # Example: select features and target
    target_col = cfg.target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Simple example: decision tree
    if cfg.pipeline.pipeline01.lower() == "decisiontree":
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        print("Training complete!")
    else:
        print(f"Pipeline {cfg.pipeline.pipeline01} not implemented.")

if __name__ == "__main__":
    main()
