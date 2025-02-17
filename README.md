### Dependency Installation
```
pip install -r requirements.txt
```
This command helps install all necessary dependencies required for the rag playground. 

**OR**

If you are using micromamba or other virtual environments,

```
micromamba create -f rag_env.yml
```

### Using chat_engine
```python
from rag_utils import create_chat_engine
chat_engine = create_chat_engine()
response = chat_engine.chat("query").response

```


