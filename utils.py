import hashlib
import json

def format_question_culturalbench_easy(data):
    question = data['prompt_question']
    options = [
        f"A. {data['prompt_option_a']}",
        f"B. {data['prompt_option_b']}",
        f"C. {data['prompt_option_c']}",
        f"D. {data['prompt_option_d']}"
    ]
    return f"{question}\n" + "\n".join(options)

def format_question_culturalbench_hard(item):
    return (f"{item['prompt_question']} "
            f"{item['prompt_option']}\n"
            f"Is this statement true or false? You must choose either True or False.")

def hash_string(s):
    """
    Generate a SHA-256 hash of the input string.
    """
    return hashlib.sha256(s.encode()).hexdigest()

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