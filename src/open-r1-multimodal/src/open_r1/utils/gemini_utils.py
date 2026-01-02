from typing import List
from PIL import Image
import os
import time
import re
import base64
import io

# Gemini model id aliasing (user can pass 'gemini', 'gemini-flash', etc.)
GEMINI_ALIAS_MAP = {
    "gemini": "gemini-2.5-flash",
    "gemini-flash": "gemini-2.5-flash",
    "gemini-2.5": "gemini-2.5-flash",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-pro-2.5": "gemini-2.5-pro",
    "gemini-2.5-pro": "gemini-2.5-pro",
}

# GPT model id aliasing
GPT_ALIAS_MAP = {
    "gpt": "gpt-5",
    "gpt-5": "gpt-5",
    "gpt-5-mini": "gpt-5-mini",
    "gpt5-mini": "gpt-5-mini",
    "gpt-mini": "gpt-5-mini",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-4-vision": "gpt-4-vision-preview",
}

# API stats for reporting
_API_STATS = {
    "gemini_retry_events": 0,
    "gemini_failed_requests": 0,
    "gemini_retry_details": [],
}

def _is_gemini_backend(model_str: str) -> bool:
    key = (model_str or "").strip().lower()
    return key.startswith("gemini")


def _resolve_gemini_model_id(model_str: str) -> str:
    key = (model_str or "").strip().lower()
    return GEMINI_ALIAS_MAP.get(key, model_str)


def _is_gpt_backend(model_str: str) -> bool:
    key = (model_str or "").strip().lower()
    return key.startswith("gpt")


def _resolve_gpt_model_id(model_str: str) -> str:
    key = (model_str or "").strip().lower()
    return GPT_ALIAS_MAP.get(key, model_str)


def _extract_mcq_choices(question: str) -> dict:
    """Extract MCQ choices from question text and return mapping {letter: option_text}.

    Supports common formats like:
    - (A) option ... (B) option ...
    - A. option ... B. option ...
    - A) option ... B) option ...
    - [A] option ... [B] option ...
    """
    text = question.replace("\n", " ")
    choices: dict[str, str] = {}
    patterns = [
        r"\(\s*([A-J])\s*\)\s*([^\(\)\[]+?)(?=\(\s*[A-J]\s*\)|\[[A-J]\]|\b[A-J]\s*[\)\.:]|$)",
        r"\b([A-J])\s*[\)\.:]\s*(.+?)(?=\b[A-J]\s*[\)\.:]|\[[A-J]\]|\(\s*[A-J]\s*\)|$)",
        r"\[([A-J])\]\s*(.+?)(?=\[[A-J]\]|\(\s*[A-J]\s*\)|\b[A-J]\s*[\)\.:]|$)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            letter = m.group(1).upper()
            option = m.group(2).strip()
            if letter not in choices and option:
                choices[letter] = option
        if len(choices) >= 2:
            break
    return choices


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (s or "").strip().lower())).strip()


def parse_mcq_letter_from_question(question: str, answer_text: str) -> str | None:
    """Return MCQ letter (A, B, C, ...) parsed from answer_text using choices in question.

    If answer_text already contains a letter, return it. Otherwise match option text.
    """
    if not question:
        return None
    # 1) Direct letter in answer (allow trailing punctuation/brackets like '.', ')', ']', '}', '>')
    ans = (answer_text or "").strip().upper()
    # exact single-letter match (with trailing punctuation removed)
    ans_clean = re.sub(r"[\s\.)\]\}>]+$", "", ans)
    if re.fullmatch(r"[A-J]", ans_clean):
        return ans_clean
    # bracketed letter variants e.g., (C), [C], {C}, <C>, possibly followed by punctuation
    bracket_pats = [
        r"\(\s*([A-J])\s*\)\s*[\.]?\s*",
        r"\[\s*([A-J])\s*\]\s*[\.]?\s*",
        r"\{\s*([A-J])\s*\}\s*[\.]?\s*",
        r"<\s*([A-J])\s*>\s*[\.]?\s*",
    ]
    for pat in bracket_pats:
        m = re.search(pat, ans)
        if m:
            return m.group(1)
    # general search anywhere (e.g., "Answer: C.")
    m = re.search(r"\b([A-J])\b", ans)
    if m:
        return m.group(1)
    # 2) Map option text to letter
    choices = _extract_mcq_choices(question)
    if not choices:
        return None
    norm_ans = _normalize(answer_text)
    if not norm_ans:
        return None
    # exact or containment match
    for letter, option in choices.items():
        norm_opt = _normalize(option)
        if not norm_opt:
            continue
        if norm_ans == norm_opt or norm_opt in norm_ans or norm_ans in norm_opt:
            return letter
    return None


_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def parse_count_number(text: str) -> str | None:
    """Parse number from text. Supports digits and common English number words."""
    if not text:
        return None
    # Prefer the first explicit integer in the text
    m = re.search(r"\b(\d{1,4})\b", text)
    if m:
        return str(int(m.group(1)))
    # Fall back to word-based numbers
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    for t in tokens:
        if t in _NUM_WORDS:
            return str(_NUM_WORDS[t])
    # Handle hyphenated like twenty-one (simple split)
    if "-" in text:
        parts = [p.strip().lower() for p in text.split("-")]
        if len(parts) == 2 and parts[0] in _NUM_WORDS and parts[1] in _NUM_WORDS:
            return str(_NUM_WORDS[parts[0]] + _NUM_WORDS[parts[1]])
    return None


def parse_yes_no(text: str) -> str | None:
    if not text:
        return None
    s = text.strip().lower()
    if re.search(r"\byes\b|\by\b|\btrue\b", s):
        return "yes"
    if re.search(r"\bno\b|\bn\b|\bfalse\b", s):
        return "no"
    return None


def normalize_verifier_answer(question: str, full_output: str, answer_text: str, question_type: str | None) -> str:
    """Normalize verifier answer by question type.

    - existence: return MCQ letter (A/B/C/...). If not found, keep original.
    - counting: return numeric string (e.g., '5'). If not found, keep original.
    - state: return 'yes' or 'no'. If not found, keep original.
    """
    qt = (question_type or "").strip().lower()
    if qt == "existence":
        letter = parse_mcq_letter_from_question(question, answer_text) or parse_mcq_letter_from_question(question, full_output)
        return letter or answer_text
    if qt == "counting":
        num = parse_count_number(answer_text) or parse_count_number(full_output)
        return num or answer_text
    if qt == "state":
        yn = parse_yes_no(answer_text) or parse_yes_no(full_output)
        return yn or answer_text
    if qt == "OCR":
        num = parse_count_number(answer_text) or parse_count_number(full_output)
        return num or answer_text
    return answer_text

def run_gemini_verifier(images: List[Image.Image] | List[str], question: str, model_id: str) -> tuple[str, str]:
    """Run verifier using Google Gemini 2.5 Flash API (new client style).

    Uses environment variable GEMINI_API_KEY.
    Returns (full_output_text, extracted_answer_text)
    """
    try:
        from google import genai
    except Exception as e:
        _API_STATS['gemini_failed_requests'] += 1
        _API_STATS['gemini_retry_details'].append(f"Missing client: {e}")
        return (f"ERROR: Missing google genai client: {e}", f"ERROR: {e}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        _API_STATS['gemini_failed_requests'] += 1
        _API_STATS['gemini_retry_details'].append("GEMINI_API_KEY not set")
        return ("ERROR: GEMINI_API_KEY not set in environment", "ERROR: GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    # Prepare inputs: client accepts PIL.Image in contents list
    pil_images: List[Image.Image] = []
    for img in images:
        if isinstance(img, str):
            pil_images.append(Image.open(img).convert('RGB'))
        elif isinstance(img, Image.Image):
            pil_images.append(img.convert('RGB'))
        else:
            _API_STATS['gemini_failed_requests'] += 1
            _API_STATS['gemini_retry_details'].append(f"Invalid image type: {type(img)}")
            return (f"ERROR: Invalid image type: {type(img)}", f"ERROR: invalid image type")

    prompt = question
    contents = [*pil_images, prompt]

    # Retry until success with exponential backoff (caps at 60s)
    attempt = 1
    sleep_s = 1
    while True:
        try:
            resp = client.models.generate_content(model=model_id, contents=contents)
            full_output = (getattr(resp, 'text', None) or "").strip()
            ans_blocks = re.findall(r'<answer>(.*?)</answer>', full_output, re.DOTALL | re.IGNORECASE)
            if ans_blocks:
                extracted = ans_blocks[-1].strip().rstrip('.').strip()
            else:
                extracted = full_output.strip().split('\n')[-1].strip().rstrip('.').strip()
            # MCQ: try force to letter using question choices
            letter = parse_mcq_letter_from_question(question, extracted) or parse_mcq_letter_from_question(question, full_output)
            if letter:
                return (full_output, letter)
            return (full_output, extracted)
        except Exception as e:
            _API_STATS['gemini_retry_events'] += 1
            msg = f"Attempt {attempt} failed: {e}"
            _API_STATS['gemini_retry_details'].append(msg)
            wait = min(sleep_s, 60)
            print(f"[Gemini] {msg}. Retrying in {wait}s...")
            time.sleep(wait)
            attempt += 1
            sleep_s = sleep_s + 1


def run_gemini_action_prediction(image: Image.Image, prompt: str, model_id: str) -> str:
    """Run action prediction using Google Gemini API.

    Uses environment variable GEMINI_API_KEY.
    Returns full_output_text from the model.
    """
    try:
        from google import genai
    except Exception as e:
        _API_STATS['gemini_failed_requests'] += 1
        _API_STATS['gemini_retry_details'].append(f"Missing client: {e}")
        return f"ERROR: Missing google genai client: {e}"

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        _API_STATS['gemini_failed_requests'] += 1
        _API_STATS['gemini_retry_details'].append("GEMINI_API_KEY not set")
        return "ERROR: GEMINI_API_KEY not set in environment"

    client = genai.Client(api_key=api_key)

    # Prepare image input
    if isinstance(image, str):
        pil_image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        pil_image = image.convert('RGB')
    else:
        _API_STATS['gemini_failed_requests'] += 1
        _API_STATS['gemini_retry_details'].append(f"Invalid image type: {type(image)}")
        return f"ERROR: Invalid image type: {type(image)}"

    contents = [pil_image, prompt]

    # Retry until success with exponential backoff (caps at 60s)
    attempt = 1
    sleep_s = 1
    while True:
        try:
            resp = client.models.generate_content(model=model_id, contents=contents)
            full_output = (getattr(resp, 'text', None) or "").strip()
            return full_output
        except Exception as e:
            _API_STATS['gemini_retry_events'] += 1
            msg = f"Attempt {attempt} failed: {e}"
            _API_STATS['gemini_retry_details'].append(msg)
            wait = min(sleep_s, 60)
            print(f"[Gemini Action] {msg}. Retrying in {wait}s...")
            time.sleep(wait)
            attempt += 1
            sleep_s = sleep_s + 1


def run_gpt_action_prediction(image: Image.Image, prompt: str, model_id: str) -> str:
    """Run action prediction using OpenAI GPT API.

    Uses environment variable OPENAI_API_KEY.
    Returns full_output_text from the model.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        _API_STATS['gemini_failed_requests'] += 1
        _API_STATS['gemini_retry_details'].append(f"Missing OpenAI client: {e}")
        return f"ERROR: Missing openai client: {e}"

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        _API_STATS['gemini_failed_requests'] += 1
        _API_STATS['gemini_retry_details'].append("OPENAI_API_KEY not set")
        return "ERROR: OPENAI_API_KEY not set in environment"

    client = OpenAI(api_key=api_key)

    # Prepare image input - convert to base64
    if isinstance(image, str):
        pil_image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        pil_image = image.convert('RGB')
    else:
        _API_STATS['gemini_failed_requests'] += 1
        _API_STATS['gemini_retry_details'].append(f"Invalid image type: {type(image)}")
        return f"ERROR: Invalid image type: {type(image)}"

    # Convert PIL image to base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Prepare input for OpenAI Responses API (following official example)
    # Note: text comes before image in the official example
    responses_input = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{img_base64}"
                }
            ]
        }
    ]

    # Retry until success with exponential backoff (caps at 60s)
    attempt = 1
    sleep_s = 1
    while True:
        try:
            # Use Responses API for GPT-5 style models
            response = client.responses.create(
                model=model_id,
                input=responses_input,
                max_output_tokens=4096
            )

            # Extract text from response following official response structure:
            # response.output[0].content[0].text
            full_output = ""
            if hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, 'type') and content_item.type == "output_text":
                                if hasattr(content_item, 'text'):
                                    full_output += content_item.text
            
            return full_output.strip()
        except Exception as e:
            _API_STATS['gemini_retry_events'] += 1
            msg = f"Attempt {attempt} failed: {e}"
            _API_STATS['gemini_retry_details'].append(msg)
            wait = min(sleep_s, 60)
            print(f"[GPT Action] {msg}. Retrying in {wait}s...")
            time.sleep(wait)
            attempt += 1
            sleep_s = sleep_s + 1