import json
import pandas as pd

def format_bbq_question(row: pd.Series) -> str:
    """
    Format a BBQ question for asking the model.
    """
    return f"{row['context']} {row['question']} (a) {row['ans0']} (b) {row['ans1']} (c) {row['ans2']}"

def extract_summary(text, question_type):
    """
    Extract the disagreement and answer from the summary JSON (which may not be well-formed).
    """
    text = text.lower()
    text = text.replace("'", '"')
    text = text.replace('\n', '').replace(' ', '')
    text = text.replace(':true', ':"true"').replace(':false', ':"false"')
    text = text.replace("'true'", '"true"').replace("'false'", '"false"')
    text = text.replace('```json', '').replace('```', '')
    if question_type == "multiple_choice":
        for option in ['A', 'B', 'C', 'D']:
            text = text.replace(f":{option}", f':"{option}"')
    text = text.replace('{disagreement', '{"disagreement"')
    text = text.replace('answer:', '"answer":')

    try:
        # Try parsing as JSON
        data = json.loads(text)
        disagreement, answer = data.get('disagreement', None), data.get('answer')
        if disagreement == 'true':
            disagreement = True
        elif disagreement == 'false':
            disagreement = False
        
        if question_type == "true_false":
            if answer == "true":
                answer = True
            elif answer == "false":
                answer = False
        return disagreement, answer
    except json.JSONDecodeError:
        # Handle plain text case
        return None, text