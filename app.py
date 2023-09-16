#!/usr/bin/env python
# coding: utf-8

# In[1]:


import openai
import os
import io
import base64 
import requests 
requests.adapters.DEFAULT_TIMEOUT = 60

import requests, json
from text_generation import Client

hf_api_key = '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'


# In[2]:


#FalcomLM-instruct endpoint on the text_generation library
#https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf
#https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct
client = Client("https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct", headers={"Authorization": f"Bearer {hf_api_key}"}, timeout=120)


# In[ ]:


#prompt = "Has math been invented or discovered?"
#client.generate(prompt, max_new_tokens=250).generated_text


# In[3]:


def format_chat_prompt(message, chat_history, system):
    prompt = f"System:{system}"
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history, system, temperature, token_number, gpt_falcon) : 
            prompt = format_chat_prompt(message, chat_history, system)
            chat_history = chat_history + [[message, ""]]
            stream = client.generate_stream(prompt,
                                              max_new_tokens=token_number,
                                              stop_sequences=["\nUser:", "<|endoftext|>"],
                                              temperature=temperature
                                               )
                                        
            acc_text = ""
            #Streaming the tokens
            for idx, response in enumerate(stream):
                    text_token = response.token.text

                    if response.details:
                        return

                    if idx == 0 and text_token.startswith(" "):
                        text_token = text_token[1:]

                    acc_text += text_token
                    last_turn = list(chat_history.pop(-1))
                    last_turn[-1] += acc_text
                    chat_history = chat_history + [last_turn]
                    yield "", chat_history
                    acc_text = ""


# In[4]:


openai.api_key = "#####"
prompt = "Enter Your Query Here"
def api_calling(prompt):
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = completions.choices[0].text
    return message
def respond_op(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output = api_calling(inp)
    history.append((input, output))
    return history, history
#block = gr.Blocks(theme=gr.themes.Monochrome())


# In[6]:


import gradio as gr

blocks = gr.Blocks()
with blocks as main_block:
    
    with gr.Tab(label = "Falcon 7B"):
        app_name = gr.Markdown("""<h2><center> Kabakoo Chatbot Falcon</center></h2>""")
        with gr.Row() as row:
            with gr.Column():
                para_mark = gr.Markdown("Parameters")
                with gr.Accordion(label="Advanced options",open=False):
                    system = gr.Textbox(label="System message", lines=2, value="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.")
                    temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.7, step=0.1)
                    token_number = gr.Slider(label="max token" , minimum =20 , maximun = 200 , value = 50 , step = 10)
                #gpt_falcon = gr.Radio(["GPT 3.5", "Falcon 7B"], label="What kind of model would you like to use?", value = "Falcon 7B")
            with gr.Column():
                chatbot = gr.Chatbot(height=240) #just to fit the notebook
                message = gr.Textbox(label="Prompt", placeholder="Prompt....", lines = 3)
                
                #gr.Examples = ["Zenith", "Antoinne", "Amelia", "Johanna"]
                with gr.Row():
                    btn = gr.Button("Submit")
                    #undo = gr.UndoButton()
                    clear = gr.ClearButton(components=[message, chatbot], value="Clear")
                    
                    btn.click(respond, inputs=[message, chatbot, system, temperature, token_number], outputs=[message, chatbot])
                    #Press enter to submit
                    #msg.submit(respond, inputs=[message, chatbot, system, temperature, token_number, gpt_falcon], outputs=[message, chatbot]) 
    with gr.Tab(label = "OpenAI"):
        gr.Markdown("""<h2><center>
         Kabakoo Chatbot OpenAI</center></h2>
        """)
        with gr.Row() as row:
            with gr.Column():
                para_mark = gr.Markdown("Parameters")
                with gr.Accordion(label="Advanced options",open=False):
                    system = gr.Textbox(label="System message", lines=2, value="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.")
                    temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.7, step=0.1)
                    token_number = gr.Slider(label="max token" , minimum =20 , maximun = 200 , value = 50 , step = 10)
            with gr.Column():
                chatbot = gr.Chatbot(height=240)
                message = gr.Textbox(placeholder="Prompt....", label  = "Prompt", lines = 3)
                state = gr.State()
                with gr.Row():
                    submit = gr.Button("Submit")
    submit.click(respond_op, inputs=[message, state], outputs=[chatbot, state])
                    
gr.close_all()       
main_block.queue().launch(share=True)

