from ...config.model_config import model_config, chat_config
from ...handle.retrieval.retrieval_utils import Search
from .prompt import naive_rag
import requests
import json


Model_Name = model_config.base_model


def post(history, is_stream=False, model_name=Model_Name):
    data = {
        'model': model_name,
        'messages': history,
        'stream': is_stream
    }
    json_data = json.dumps(data)
    if model_config.is_online:
        response = requests.post(
            chat_config['online'][model_name]['url'],
            data=json_data,
            headers=chat_config['online'][model_name]['headers'],
            timeout=300,
            stream=is_stream
        )
    else:
        response = requests.post(
            'https://{ip}:{port}/v1/chat/completions'.format(ip=chat_config['local'][model_name]['ip'], port=str(chat_config['local'][model_name]['port'])),
            data=json_data,
            timeout=300,
            stream=is_stream
        )
        # print(response)
    return response


def model_message(query, history=[], is_stream=False, model_name=Model_Name):
    history.append(
        {
            'role': 'user',
            'content': query
        }
    )
    response = post(history, is_stream=is_stream, model_name=model_name)
    if is_stream:
        history.append({'role': 'assistant', 'content': ''})
        for line in response.iter_lines(decode_unicode=True):
            if 'data: ' in line:
                if line.replace('data: ', '') == '[DONE]':
                    break
                else:
                    result = json.loads(line.replace('data: ', ''))['choices'][0]['delta']
                    if not result.get('content'):
                        continue
                    history[-1]['content'] += result['content']
                    yield result['content'], history
    else:
        result = response.json()['choices'][0]['message']
        history.append(result)
        yield result['content'], history


def chat(query, history=[], retriever=None, is_stream=False, model_name=Model_Name, reranker=None):
    if retriever:
        content = retriever.search_for_content(query)
        if reranker:
            content = reranker(content)
        content = '\n\n'.join(content)
        if is_stream:
            return model_message(naive_rag.format(content=content, query=query), history=history, is_stream=is_stream, model_name=model_name)
        else:
            response, history = next(model_message(naive_rag.format(content=content, query=query), history=history, is_stream=is_stream, model_name=model_name))
            return response, history
    else:
        if is_stream:
            return model_message(query, history=history, is_stream=is_stream, model_name=model_name)
        else:
            response, history = next(model_message(query, history=history, is_stream=is_stream, model_name=model_name))
            return response, history
