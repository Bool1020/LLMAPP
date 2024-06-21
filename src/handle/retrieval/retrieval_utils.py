from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.tools import DuckDuckGoSearchRun


def load_docs(path):
    if path.endswith('.txt'):
        loader = TextLoader(path)
    elif path.endswith('.pdf'):
        loader = PyPDFLoader(path)
    elif path.endswith('.docx') or path.endswith('.doc'):
        loader = Docx2txtLoader(path)
    return loader.load()

class Search(object):
    def __init__(self, db, top_k=5):
        self.retriever = db.as_retriever(search_kwargs={"k": top_k})

    def __call__(self, query):
        return self.retriever.get_relevant_documents(query)

    def search_for_content(self, query):
        docs = self(query)
        for i, doc in enumerate(docs):
            docs[i] = doc.page_content
        return docs

    def search_for_qa(self, query):
        docs = self(query)
        for i, doc in enumerate(docs):
            docs[i] = {'question': doc.page_content, 'answer': doc.metadata['answer']}
        return docs

    def search_for_docs(self, query):
        docs = self(query)
        docs_set = set()
        for i, doc in enumerate(docs):
            docs_set.add(doc.metadata['source'])
        return docs_set
    
    def auto_merge(self, query, therehold=0.5):
        docs = self(query)
        sta = {}
        total = {}
        for doc in docs:
            total[doc.metadata['source']] = doc.metadata['total']
            if doc.metadata['source'] in sta:
                sta[doc.metadata['source']] += 1
            else:
                sta[doc.metadata['source']] = 1
        for source in sta:
            if sta[source] > total[source] * therehold:
                docs = list(filter(lambda x: x.metadata['source'] != source, docs))
                docs.append(load_docs(source))
        for i, doc in enumerate(docs):
            docs[i] = doc.page_content
        return docs
    
    def web_search(self, query):
        search = DuckDuckGoSearchRun()
        return[search.run(query)]
