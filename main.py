import gradio as gr
import os
import json
from src.handle.chat.api_rpc import chat
from src.handle.knowledge.knowledge_utils import load_knowledge
from src.handle.retrieval.retrieval_utils import Search
from src.config.model_config import chat_config
from datetime import datetime

# 获取当前时间
now = datetime.now()
time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
db = None
retriever = None
reranker = None

history_list = [time_str]
model_list = []
knowledge_list = []

for history in os.listdir('history'):
    history_list.append(history.replace('.json', ''))
for model_name in chat_config['online']:
    model_list.append(model_name)
for knowledge_name in os.listdir('knowledge_base/content'):
    knowledge_list.append(knowledge_name)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            model_name = gr.Dropdown(choices=model_list, value=model_list[0], label="Model")
            history_name = gr.Dropdown(choices=history_list, value=time_str, label="History")
            knowledge_name = gr.Dropdown(choices=knowledge_list, value=None, label="Knowledge")
            num_chunk = gr.Slider(minimum=1, maximum=20, value=5, scale=1, label="number of chunk")

            def knowledge_change(knowledge, num):
                global retriever
                global db
                db = load_knowledge(knowledge)
                retriever = Search(db, num)

            def num_change(num):
                global db
                global retriever
                retriever = Search(db, num)

            knowledge_name.change(knowledge_change, [knowledge_name, num_chunk], [])
            num_chunk.change(num_change, [num_chunk], [])


        with gr.Column(scale=100):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(show_label=False)


            def history_change(file_path, his):
                global time_str
                if his != []:
                    with open('history/' + time_str + '.json', 'w', encoding='utf-8') as f:
                        json.dump(his, f, ensure_ascii=False, indent=4)
                time_str = file_path
                try:
                    with open('history/' + file_path + '.json', 'r', encoding='utf-8') as f:
                        his = json.load(f)
                except:
                    his = []
                return '', his


            def new_chat():
                now = datetime.now()
                time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
                new_history = time_str
                history_list.append(new_history)
                return gr.update(choices=history_list, value=new_history)


            new = gr.Button('New')


            def respond(message, history, model_name):
                chat_history = []
                if len(history) > 0:
                    for his in history:
                        chat_history += [{'role': 'user', 'content': his[0]}, {'role': 'assistant', 'content': his[1]}]
                response = chat(message, chat_history, retriever=retriever, is_stream=True, model_name=model_name)
                history.append((message, ''))
                for _, his in response:
                    history[-1] = (message, his[-1]['content'])
                    yield '', history



            new.click(new_chat, [], [history_name])
            history_name.change(history_change, [history_name, chatbot], [msg, chatbot])
            msg.submit(respond, [msg, chatbot, model_name], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()