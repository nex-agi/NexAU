# Multimodal Image Handling in NexAU

This document explains how to use **Image Inputs** (sending images to an agent) and **Image Tool Results** (returning images from tools) within the NexAU framework.

## 1. Sending Images to the Agent (Image Input)

You can provide images to an agent as part of the user message. The framework supports both **URLs** and **Base64 Data URIs**.

### Using `Message` and `ImageBlock`

This approach provides strong typing and is the most robust way to construct messages programmatically.

```python
import base64
from nexau.core.messages import Message, Role, TextBlock, ImageBlock

# Option 1: Using a remote URL
msg_url = Message(
    role=Role.USER,
    content=[
        TextBlock(text="Please describe this chart."),
        ImageBlock(url="https://example.com/chart.png")
    ]
)

# Option 2: Using Base64 Data (for local files)
with open("local_screenshot.png", "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

msg_base64 = Message(
    role=Role.USER,
    content=[
        TextBlock(text="Analyze this screenshot."),
        # Ensure you provide the correct mime_type
        ImageBlock(base64=b64_data, mime_type="image/png")
    ]
)

agent = Agent(config=AgentConfig(...))
# Sending the message to the agent
response = agent.run([msg_base64])

```

---

## 2. Returning Images from Tools (Image Tool Results)

Tools can return images directly to the agent. This is useful for tools that generate charts, capture screenshots, or process visual files. The agent "sees" these images just as if the user had uploaded them.

### Method A: Returning a Dictionary (Easiest)

Your tool function can simply return a dictionary with specific keys (`type`, `image_url`, or `base64`).

**Example: Tool returning a generated chart**

```python
import base64
import matplotlib.pyplot as plt
import io

def generate_sales_chart(data: list[int]) -> dict:
    """Generates a sales chart and returns it as an image."""
    plt.plot(data)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")

    # Return as a Data URI (Standard OpenAI format)
    return {
        "type": "image",
        "image_url": f"data:image/png;base64,{b64_str}",
        "detail": "auto"
    }

```

### Method B: Using `ToolOutputImage` Helper

For better type safety and clarity, use the `ToolOutputImage` class from `nexau.core`.

```python
from nexau.core.messages import ToolOutputImage

def get_webcam_snapshot() -> ToolOutputImage:
    """Captures a webcam frame."""
    # ... logic to capture frame ...
    image_url = "https://internal-storage/snapshot-123.jpg"

    return ToolOutputImage(
        image_url=image_url,
        detail="low"
    )

```

### Method C: Returning Mixed Content (Text + Image)

If your tool needs to return both a textual explanation and an image, return a **list**.

```python
def analyze_csv_visualize(csv_path: str) -> list:
    # ... processing logic ...

    return [
        {"type": "text", "text": "I have visualized the data from the CSV file:"},
        {
            "type": "image",
            "image_url": "data:image/png;base64,..."
        }
    ]

```

---

## 3. Key Considerations

| Feature | Note |
| --- | --- |
| **Model Support** | Ensure your LLM supports vision (e.g., `gpt-4o`). Non-vision models will ignore image blocks or only receive fallback text. |
| **Tokens & Cost** | Images consume significantly more tokens than text. Use `detail="low"` for images where high resolution is not required to reduce costs. |
| **Platform Compatibility** | NexAU automatically handles platform differences. For example, OpenAI Chat Completions do not natively support images in `tool` messages; NexAU automatically injects them as user messages behind the scenes. |

### Reference: Image Object Fields

| Field | Description |
| --- | --- |
| `type` | Must be `"image"`, `"image_url"`, or `"input_image"`. |
| `image_url` | A URL string (`http...`) or a Data URI (`data:image/png;base64,...`). |
| `base64` | Raw base64 string (without the `data:` prefix). Used if `image_url` is not provided. |
| `detail` | `"auto"` (default), `"low"`, or `"high"`. Controls how the model processes image resolution. |
