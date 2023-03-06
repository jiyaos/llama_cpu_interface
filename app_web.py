import os
import time
import torch
import gradio as gr


from gen_web import get_pretrained_models, get_output, setup_model_parallel

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"


local_rank, world_size = setup_model_parallel()
generator = get_pretrained_models("7B", "tokenizer", local_rank, world_size)

history = []
simple_history = []


def chat(user_input, top_p, temperature, max_gen_len):
    bot_response = get_output(
        generator=generator,
        prompt=user_input,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p)

    # remove the first phrase identical to user prompt
    bot_response = bot_response[0][len(user_input):]
    # trip the last phrase
    try:
        bot_response = bot_response[:bot_response.rfind(".")]
    except:
        pass

    history.append({
        "role": "user",
        "content": user_input
    })
    history.append({
        "role": "system",
        "content": bot_response
    })

    simple_history.append((user_input, None))

    response = ""
    for word in bot_response.split(" "):
        time.sleep(0.1)
        response += word + " "
        current_pair = (user_input, response)
        simple_history[-1] = current_pair
        yield simple_history


def reset_textbox():
    return gr.update(value='')


with gr.Blocks(css="""#col_container {width: 85%; margin-left: auto; margin-right: auto;}
                #chatbot {height: 500px; overflow: auto;}""") as demo:
    with gr.Column(elem_id='col_container'):
        gr.Markdown(" ## LLaMA 7B Model ")
        chatbot = gr.Chatbot(elem_id='chatbot')
        textbox = gr.Textbox(placeholder="Enter a prompt")

        with gr.Accordion("Parameters", open=False):
            max_gen_len = gr.Slider(minimum=20, maximum=30, value=256, step=1, interactive=True,
                                    label="Max Genenration Length", )
            top_p = gr.Slider(minimum=-0, maximum=1.0, value=1.0, step=0.05, interactive=True,
                              label="Top-p (nucleus sampling)", )
            temperature = gr.Slider(minimum=-0, maximum=5.0, value=1.0, step=0.1, interactive=True,
                                    label="Temperature", )

    textbox.submit(chat, [textbox, top_p, temperature, max_gen_len], chatbot)
    textbox.submit(reset_textbox, [], [textbox])

demo.queue(api_open=False).launch()