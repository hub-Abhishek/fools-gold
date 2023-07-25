from chromadb.api.types import EmbeddingFunction
from typing import Any, Optional, Union, Dict, Sequence, TypeVar, List
from tqdm import tqdm

Number = Union[int, float]
Embedding = List[Number]
Embeddings = List[Embedding]


Metadata = Dict[str, Union[str, int, float]]
Metadatas = List[Metadata]

CollectionMetadata = Dict[Any, Any]

Document = str
Documents = List[Document]

class embeddings_function(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        ...

    def call_url(self, texts: Documents):
        url = 'https://neo63xnfpd.execute-api.us-east-1.amazonaws.com/dev/bot-api' # f"{secrets['model']['url']}"
        return requests.post(data=json.dumps({'data': texts, 'decode_level': 'embed'}), url=url).json()
        ...
    def embed_documents(self, texts: List) -> Embeddings:
        # url = f"{secrets['model']['url']}"
        if len(texts) <= 5:
            returned_json = self.call_url(texts)
            return eval(returned_json)
        else:
            all_returned_json = []
            for i in tqdm(range(0, len(texts), 5)):
                returned_json = self.call_url(texts[i:i+5])
                print(returned_json)
                all_returned_json += eval(returned_json)
            return all_returned_json
        
    
    def embed_query(self, text: str) -> List[float]:
        _texts = [text]
        return self.embed_documents(_texts)[0]
    


from typing import Any, List, Mapping, Optional
import requests, json

class CustomLLM:
    # n: int
    def __init__(self, url) -> None:
        
        self.url = url

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        documents: Optional[Documents] = None,
        decode_level: str = "generate",
        # stop: Optional[List[str]] = None,
        # run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        url = self.url # f"{secrets['model']['url']}"
        returned_json = requests.post(data=json.dumps({'data': prompt, 'documents': documents, 'decode_level': decode_level}), url=url).json()
        return eval(returned_json)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        # return {"n": self.n}
        return {}