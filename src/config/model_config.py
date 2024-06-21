api_key = {
    'baichuan': 'sk-74a4c635c2f59615b2e5b044f9dcf64e',
    'deepseek': 'sk-e4d13bee26684e8fad6fbf9c286069d3'
}


chat_config = {
    'local': {
        'Baichuan2-13B-Chat': {
            'ip': '0.0.0.0',
            'port': 5001,
        },
        'Qwen-14B-Chat': {
            'ip': '0.0.0.0',
            'port': 5001,
        },
        'chatglm3-6b': {
            'ip': '0.0.0.0',
            'port': 5001,
        },
        'Baichuan2-7B-Chat': {
            'ip': 'u199040-a426-29f8b9a8.westc.gpuhub.com',
            'port': 8443,
        },
    },
    'online': {
        'Baichuan2-Turbo': {
            'url': 'https://api.baichuan-ai.com/v1/chat/completions',
            'headers': {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + api_key['baichuan']
            }
        },
        'Baichuan2-Turbo-192k': {
            'url': 'https://api.baichuan-ai.com/v1/chat/completions',
            'headers': {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + api_key['baichuan']
            }
        },
        'Baichuan2-53B': {
            'url': 'https://api.baichuan-ai.com/v1/chat/completions',
            'headers': {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + api_key['baichuan']
            }
        },
        'deepseek-chat': {
            'url': 'https://api.deepseek.com/v1/chat/completions',
            'headers': {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + api_key['deepseek']
            }
        }
    }
}


class ModelConfig:
    embedding_model = 'bge-m3'
    base_model = 'Baichuan2-7B-Chat'
    is_online = base_model in chat_config['online']


model_config = ModelConfig()
