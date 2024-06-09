import torch


class Reranker(object):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

    def compute_score(self, pairs):
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
        return scores

    def __call__(self, query, content):
        pairs = []
        for i in range(len(content)):
            pairs.append([query, content[i]])
        scores = self.compute_score(content)
        combined = list(zip(content, scores))
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        sorted_list, sorted_scores = zip(*sorted_combined)

        return list(sorted_list)

