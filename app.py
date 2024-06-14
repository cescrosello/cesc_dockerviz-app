import os
import re

import numpy as np
import panel as pn
import tastymap as tm
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(".env", override=True)

# Configuration
COLORMAP = "cool"
COLORMAP_REVERSE = False
NUM_COLORS = 10
SYSTEM_PROMPT = ""  # Keep empty for user-provided prompts
TOP_LOGPROBS = 5

# Get OpenAI API key (Prioritize environment variable for security)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
pn.extension()


def color_by_logprob(text, log_prob):
    """Calculates color index and generates styled HTML."""
    linear_prob = np.round(np.exp(log_prob) * 100, 2)
    color_index = int(linear_prob // (100 / (len(colors) - 1)))

    # Simpler HTML formatting
    html_output = f'<span style="color: {colors[color_index]}"><b>{text}</span>'
    return html_output


def custom_serializer(content):
    """Extracts plain text from HTML for memory."""
    pattern = r"<span.*?>(.*?)</span>"
    matches = re.findall(pattern, content)
    return matches[0] if matches else content


async def respond_to_input(contents: str, user: str, instance: pn.chat.ChatInterface):
    """Handles user input, calls OpenAI, and formats the response."""
    if api_key_input.value:
        aclient.api_key = api_key_input.value
    elif not os.environ["OPENAI_API_KEY"]:
        instance.send("Please provide an OpenAI API key", respond=False, user="ChatGPT")

    messages = []
    if system_input.value:
        messages.append({"role": "system", "content": system_input.value})
    messages.append({"role": "user", "content": contents})

    if memory_toggle.value:
        messages += instance.serialize(custom_serializer=custom_serializer)

    try:
        response = await aclient.chat.completions.create(
            model=model_selector.value,
            messages=messages,
            top_logprobs=TOP_LOGPROBS,
            stream=True,
            logprobs=True,
            temperature=temperature_input.value,
            max_tokens=max_tokens_input.value,
            seed=seed_input.value,
        )

        message = ""
        cosa = """<br><strong><p style="color:dark-blue">"""
        async for chunk in response:            
            choice = chunk.choices[0]
            ps=[]
            try:
                ps = [(x.token,np.round(np.exp(x.logprob) * 100, 2)) 
                    for x in chunk.choices[0].logprobs.content[0].top_logprobs]
                print(ps)
            except:
                pass
            content = choice.delta.content
            log_probs = choice.logprobs

            if content and log_probs:
                log_prob = log_probs.content[0].logprob
                message += color_by_logprob(content, log_prob)
                message += "&nbsp;"*(20-len(content))+ str(dict(ps)) + "<br>"
                cosa += content
                yield message + cosa+"</strong>"

    except Exception as e:
        instance.send(f"An error occurred: {e}", respond=False, user="ChatGPT")


# Colormap generation 
tmap = tm.cook_tmap(COLORMAP, NUM_COLORS, COLORMAP_REVERSE)
colors = tmap.to_model("hex")

# OpenAI client setup
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

# UI Widgets
api_key_input = pn.widgets.PasswordInput(
    name="OpenAi API Key",
    placeholder=OPENAI_API_KEY,
    width=150,
)
system_input = pn.widgets.TextAreaInput(
    name="System Prompt", value=SYSTEM_PROMPT, rows=2, auto_grow=True
)
model_selector = pn.widgets.Select(
    name="Model Selector", options=["gpt-3.5-turbo", "gpt-4"], width=150
)
temperature_input = pn.widgets.FloatInput(
    name="Temperature", start=0, end=4, step=0.01, value=1, width=100
)
max_tokens_input = pn.widgets.IntInput(
    name="Max Tokens", start=0, value=20, width=100
)
seed_input = pn.widgets.IntInput(name="Seed", start=0, end=100, value=0, width=100)
memory_toggle = pn.widgets.Toggle(name="Include Memory", value=False, margin=(10, 5))
chat_interface = pn.chat.ChatInterface(
    callback=respond_to_input, 
    callback_user="ChatGPT", 
    callback_exception="verbose", 
    max_width=1300
)

# Colormap Bar Visualization
bar = f"""
<div style="height: 8px; padding-left: 700px;">
    0% - (Color intensity indicates token likelihood) - 100% 

{re.findall(r'<div class="cmap".*?</div>', tmap._repr_html_())[0]} </div>
"""
bar = re.sub(r'" src="', 'max-width: 320px; " src="', bar)

# Panel Template Setup
template = pn.template.MaterialTemplate(title='~ToKEn viSuAl~')
template.main.append(
    pn.Column(
        pn.Row(
            memory_toggle, system_input, model_selector,
            temperature_input, max_tokens_input, seed_input, 
            api_key_input, align="center"
        ),
        pn.Row(bar, align="center"),
        chat_interface,
    )
)

template.show(port=60000)

