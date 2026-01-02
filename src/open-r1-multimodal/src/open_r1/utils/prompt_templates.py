ACTION_PROMPT_TEMPLATE = (
    "You are an embodied agent navigating a 3D scene from an egocentric camera.\n"
    "Given the current image and a question about the scene, predict the optimal NEXT action parameters to reach a better viewpoint.\n"
    "\n"
    "Action parameters (return integers only):\n"
    "1) Heading rotation (deg) in (-180, 180]: Azimuth yaw about your vertical axis BEFORE moving.\n"
    "   Positive = clockwise/right, negative = counterclockwise/left, 0 = no rotation.\n"
    "2) Forward distance (cm) >= 0: Move forward in the NEW facing direction after the rotation. 0 = no move.\n"
    "3) View rotation (deg) in (-180, 180]: Final azimuth adjustment AFTER moving, relative to your post-move heading.\n"
    "   Same sign convention as rotation.\n"
    "\n"
    "Goal: Choose a heading rotation angle, moving forward distance, final-viewing rotation angle that maximizes visibility of task-relevant objects and minimizes occlusion.\n"
    "Example: Rotating -90 degrees, moving forward 50 cm, then rotating 90 degrees is equivalent to translating 50 cm to your left while keeping the original heading.\n"
    "Question: {question}\n"
    "DO NOT answer the question; ONLY predict the next action parameters.\n"
)

# RL Format Prompt: single thinking process followed by all action tags
GRPO_FORMAT_PROMPT = (
    "First output the reasoning process in <think> </think> tags.\n"
    "Then, output the final predictions in <head> </head>, <fwd> </fwd>, <view> </view> tags in order.\n"
    "The text between <head> and </head> must be the angle in degrees (-180, 180], <fwd> and </fwd> must be the nonnegative forward distance, and <view> and </view> must be the final viewing angle in degrees (-180, 180].\n"
    "Each must be exactly one integer number (no units, no extra text).\n"
    "In the reasoning process, explicitly reason about (1) how much to rotate to determine the moving direction, (2) how far to move forward to approach, (3) how much to further adjust your azimuth angle from your moving direction for the best view.\n"
    "\n"
)

# SFT-then-RL Format Prompt: initial guess, reasoning, then final guess
SFT_GRPO_FORMAT_PROMPT = (
    "First, output your initial guess for the action parameter values.\n"
    "Then, think carefully to refine your initial guess for each action parameter.\n"
    "After output initial guess for the action parameters, output your reasoning process within <think> </think> tags, and then provide the final guess within <head> </head>, <fwd> </fwd>, and <view> </view> tags, respectively.\n"
    "The text between <head> and </head> must be the rotation angle in degrees in the range (-180, 180]; the text between <fwd> and </fwd> must be the nonnegative forward distance; and the text between <view> and </view> must be the final viewing angle in degrees in the range (-180, 180].\n"
    "Each must be exactly one integer (no units, no extra text).\n"
    "In the reasoning process, explicitly reason about (1) how much to rotate to determine the moving direction, (2) how far to move forward to approach, (3) how much to further adjust your azimuth angle from your moving direction for the best view.\n"
    "\n"
    "For example:\n"
    "<head> INITIAL_GUESS </head> <fwd> INITIAL_GUESS </fwd> <view> INITIAL_GUESS </view>\n"
    "<think> REASONING PROCESS </think>\n"
    "<head> FINAL_GUESS </head> <fwd> FINAL_GUESS </fwd> <view> FINAL_GUESS </view>\n"
)

# SFT Format Prompt: direct output without reasoning
SFT_FORMAT_PROMPT = (
    "Output the final predictions in <head> </head>, <fwd> </fwd>, <view> </view> tags in order.\n"
    "The text between <head> and </head> must be the rotation angle in degrees (-180, 180], <fwd> and </fwd> must be the nonnegative forward distance, and <view> and </view> must be the final viewing angle in degrees (-180, 180].\n"
    "Each must be exactly one integer number (no units, no extra text).\n"
)
