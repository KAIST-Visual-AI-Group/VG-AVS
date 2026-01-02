import re

def clean_text(text, exclue_chars=["\n", "\r"]):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]

    for char in exclue_chars:
        if char in ["\n", "\r"]:
            # If there is a space before the newline, remove the newline
            text = re.sub(r"(?<=\s)" + re.escape(char), "", text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r"(?<!\s)" + re.escape(char), " ", text)
        else:
            text = text.replace(char, " ")

    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip(".").lower()


def extract_choice(text):
    # 1. Clean and normalize text
    text = text.upper()  # Convert to uppercase
    text = re.sub(r"\s+", " ", text)  # Normalize spaces

    # 2. Choice should not have uppercase letters before or after
    choices = re.findall(r"(?<![A-Z])([A-Z])(?=[\.\\,\?\!\:\;]|$)", text)

    if not choices:
        return None

    # 3. If only one choice, return it directly
    if len(choices) == 1:
        return choices[0]

    # 4. If multiple choices, use heuristic rules
    choice_scores = {choice: 0 for choice in choices}

    # 4.1 Keywords around choices get points
    keywords = [
        "answer",
        "correct",
        "choose",
        "select",
        "right",
        "think",
        "believe",
        "should",
    ]

    # Get context for each choice (20 chars before and after)
    for choice in choices:
        pos = text.find(choice)
        context = text[max(0, pos - 20) : min(len(text), pos + 20)]

        # Add points for keywords
        for keyword in keywords:
            if keyword.upper() in context:
                choice_scores[choice] += 1

        # Add points if choice is near the end (usually final answer)
        if pos > len(text) * 0.7:  # In last 30% of text
            choice_scores[choice] += 2

        # Add points if followed by punctuation
        if pos < len(text) - 1 and text[pos + 1] in "。.!！,，":
            choice_scores[choice] += 1

    # Return highest scoring choice
    return max(choice_scores.items(), key=lambda x: x[1])[0]


def action_string_to_list(text: str):
    """Extract three integers from the model output using special tags: <head>, <fwd>, <view>.

    Rules:
      - Extracts values from <head>...</head>, <fwd>...</fwd>, <view>...</view> tags
      - Returns a list of three integers: (1) heading rotation (degrees), (2) forward distance (centimeters), (3) view rotation (degrees)
      - If any tag is missing or invalid, uses 0 as default
    """
    # Try to extract any integer or float value (including negatives) from special tags
    head_match = re.search(r'<head>\s*(-?\d*\.?\d+)\s*</head>', text, re.IGNORECASE)
    fwd_match = re.search(r'<fwd>\s*(-?\d*\.?\d+)\s*</fwd>', text, re.IGNORECASE)
    view_match = re.search(r'<view>\s*(-?\d*\.?\d+)\s*</view>', text, re.IGNORECASE)

    # Extract values, default to 0 if not found
    head = float(head_match.group(1)) if head_match else 0
    fwd = float(fwd_match.group(1)) if fwd_match else 0
    view = float(view_match.group(1)) if view_match else 0
    
    return [head, fwd, view]

def extract_thinking_and_action_string(text: str) -> tuple[str, str]:
    """Extract thinking process and action tags separately.
    
    Note: Extracts the LAST <think> tag content since the output may contain
    the prompt with examples that also have <think> tags.
    """
    # Extract ALL <think> content matches and use the LAST one
    think_matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    if think_matches:
        thinking = think_matches[-1].strip()  # Use the LAST match
    else:
        thinking = "N/A"
    
    # Extract action tags (use last match for each as well)
    # Updated regex: capture any integer or float (positive/negative, with/without decimals, optional whitespace)
    head_matches = re.findall(r'<head>\s*(-?\d+(?:\.\d+)?|\.\d+)\s*</head>', text, re.IGNORECASE)
    fwd_matches = re.findall(r'<fwd>\s*(-?\d+(?:\.\d+)?|\.\d+)\s*</fwd>', text, re.IGNORECASE)
    view_matches = re.findall(r'<view>\s*(-?\d+(?:\.\d+)?|\.\d+)\s*</view>', text, re.IGNORECASE)

    head = head_matches[-1] if head_matches else "N/A"
    fwd = fwd_matches[-1] if fwd_matches else "N/A"
    view = view_matches[-1] if view_matches else "N/A"
    
    actions_text = f"<head>{head}</head> <fwd>{fwd}</fwd> <view>{view}</view>"
    
    return thinking, actions_text
