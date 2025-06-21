from pathlib import Path
import pandas as pd

def format_bbq_question(row: pd.Series) -> str:
    """
    Format a BBQ question for asking the model.
    """
    return f"{row['context']} {row['question']} (a) {row['ans0']} (b) {row['ans1']} (c) {row['ans2']}"

def load_bbq_df(categories: list[str]):
    dfs = []
    for category in categories:
        data_path = Path('data/bbq') / f"{category}.jsonl"

        # Load dataset
        df = pd.read_json(data_path, lines=True)
        df = df[df['context_condition'] == "ambig"]
        df = df.drop(columns=["question_index", "context_condition", "additional_metadata", "answer_info"])
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    # Merge with additional metadata to get the target_loc (the index of the answer option that corresponds to the bias target)
    df = df.merge(pd.read_csv('data/bbq/additional_metadata.csv', usecols=['category', 'example_id', 'target_loc']), on=['category', 'example_id'])
    df = df[~df['target_loc'].isnull()] # remove some examples that don't have target_loc
    df['target_loc'] = df['target_loc'].astype(int)
    df = df.rename(columns={'target_loc': 'biased_ans_label'})
    df = df.reset_index(drop=True)
    
    return df