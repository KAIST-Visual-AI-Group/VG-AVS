from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional
import os
from PIL import Image, ImageDraw, ImageFont
import textwrap


def visualize(primary_img, add_view, actions, question, verifier_ans, sol):
    pad = 24
    gap = 16
    text_gap = 16
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    header = f"Q: {question}\nActions: {actions}\nVerifier: {verifier_ans}\nSol: {sol}"

    # Calculate text size
    tmp = Image.new("RGB", (10, 10), "white")
    dtmp = ImageDraw.Draw(tmp)
    text_bbox = dtmp.multiline_textbbox((0, 0), header, font=font, spacing=4)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    # Canvas size
    combo_w = primary_img.width + gap + add_view.width
    canvas_w = max(combo_w, text_w) + pad * 2
    canvas_h = pad + text_h + text_gap + primary_img.height + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Text
    draw.multiline_text((pad, pad), header, fill=(0, 0, 0), font=font, spacing=4)

    # Image placement (left: p, right: a)
    y0 = pad + text_h + text_gap
    x_left = (canvas_w - combo_w) // 2
    canvas.paste(primary_img, (x_left, y0))
    canvas.paste(add_view, (x_left + primary_img.width + gap, y0))

    return canvas


def wrap_text(text: str, width: int = 60) -> str:
    """Wrap text to specified width."""
    return "\n".join(textwrap.wrap(text, width=width))


def create_visualization(
    input_img_or_path: Image.Image | str,
    generated_img_or_path: Image.Image | str,
    predicted_actions: List[int],
    gt_actions: List[int],
    action_question: str,
    vqa_question: str,
    thinking_process: str,
    verifier_answer: str,
    gt_answer: str,
    is_correct: bool,
    output_path: str,
    llm_score: Optional[float] = None,
    gt_img_or_path: Image.Image | str = None,
    actions_text: Optional[str] = None,
):
    """Create visualization combining all information."""
    if isinstance(input_img_or_path, str):
        img_input = Image.open(input_img_or_path).convert("RGB")
    else:
        img_input = input_img_or_path.convert("RGB")

    # Generated image may be None (e.g., generation failed) – fallback to input image
    if generated_img_or_path is None:
        img_generated = img_input.copy()
    elif isinstance(generated_img_or_path, str):
        img_generated = Image.open(generated_img_or_path).convert("RGB")
    elif isinstance(generated_img_or_path, Image.Image):
        img_generated = generated_img_or_path.convert("RGB")
    else:
        img_generated = Image.new("RGB", img_input.size, color=(128, 128, 128))

    if gt_img_or_path is None:
        img_gt = None
    elif isinstance(gt_img_or_path, str):
        img_gt = Image.open(gt_img_or_path).convert("RGB")
    else:
        img_gt = gt_img_or_path.convert("RGB")

    # Create canvas
    img_w, img_h = img_input.size
    
    text_panel_width = 600
    total_width = img_w * 2 + text_panel_width
    if img_gt is not None: total_width += img_w
    total_height = max(img_h, 1200)  # Increased height to prevent text truncation

    canvas = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    
    # paste images
    canvas.paste(img_input, (0, 0))
    canvas.paste(img_generated, (img_w, 0))
    if img_gt is not None:
        canvas.paste(img_gt, (img_w * 2, 0))

    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font_title = ImageFont.load_default()
        font_normal = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    text_x = img_w * 2 + 20
    if img_gt is not None: text_x += img_w
    text_y = 20
    line_spacing = 22

    draw.text((10, img_h + 10), "Input View", fill=(0, 0, 0), font=font_normal)
    draw.text((img_w + 10, img_h + 10), "Generated View (predicted)", fill=(0, 0, 0), font=font_normal)
    if img_gt is not None:
        draw.text((img_w * 2 + 10, img_h + 10), "Ground Truth View", fill=(0, 0, 0), font=font_normal)


    # Write text info
    def write_section(title, content, color=(0, 0, 0), max_lines=None, title_font=None, content_font=None):
        nonlocal text_y
        title_font = font_title if title_font is None else title_font
        content_font = font_small if content_font is None else content_font
        draw.text((text_x, text_y), title, fill=color, font=title_font)
        text_y += line_spacing + 3
        
        # Skip content rendering if empty
        if content and content.strip():
            wrapped = wrap_text(str(content), width=80)
            line_count = 0
            for line in wrapped.split('\n'):
                if text_y > total_height - 50:
                    draw.text((text_x, text_y), "... (truncated)", fill=(100, 100, 100), font=content_font)
                    text_y += 15
                    break
                if max_lines and line_count >= max_lines:
                    draw.text((text_x, text_y), "... (truncated)", fill=(100, 100, 100), font=content_font)
                    text_y += 15
                    break
                draw.text((text_x, text_y), line, fill=(50, 50, 50), font=content_font)
                text_y += 16
                line_count += 1
        text_y += 12
    
    def write_single_line(text, color=(0, 0, 0)):
        """Write a single line without title."""
        nonlocal text_y
        draw.text((text_x, text_y), text, fill=color, font=font_small)
        text_y += 18
    
    write_section("Action Question:", action_question, color=(150, 150, 0), content_font=font_small)
    write_section("VQA Question:", vqa_question, color=(0, 0, 150), content_font=font_small)
    write_section("Thinking Process:", thinking_process, color=(100, 0, 100), content_font=font_normal)
    
    # Use actions_text if provided, otherwise format predicted_actions
    if actions_text is not None:
        predicted_actions_str = actions_text
    else:
        predicted_actions_str = f"head={predicted_actions[0]:.2f}°, fwd={predicted_actions[1]:.2f}cm, view={predicted_actions[2]:.2f}°"
    
    write_section("Predicted Actions:", predicted_actions_str, color=(150, 0, 0), content_font=font_normal)
    # GT actions are not displayed as requested
    
    write_section("Verifier Answer:", verifier_answer, color=(0, 150, 150), content_font=font_normal)
    write_section("GT Answer:", gt_answer, color=(0, 0, 0), content_font=font_normal)

    if llm_score is not None:
        write_single_line(f"LLM Score: {llm_score}/5", color=(100, 100, 100))
    else:
        accuracy_text = "✓ CORRECT" if is_correct else "✗ WRONG"
        accuracy_color = (0, 200, 0) if is_correct else (200, 0, 0)
        draw.text((text_x, text_y), accuracy_text, fill=accuracy_color, font=font_title)


    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    canvas.save(output_path)
    print(f"Saved visualization: {output_path}")