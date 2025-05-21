# MA Katrin Ibrahim

## Setup

Create a virtual environment, activate it, install the dependencies, and add the project to the Python path:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:./
export HF_TOKEN=<your Hugging Face API token>
```
## TODO
- [ ] clean up configs (move to folder)
- [ ] make model name configurable
- [ ] text size configurable
- [ ] prompt template for writer
- [ ] evaluator prompt template
- [ ] add web search knowledge source
- [ ] explore scientific articles knowledge source
- [ ] add user goal
- [ ] testing loop (10-20 wildseek topics)
- [ ] organize utils
- [ ] add a unified logging system
- [ ] create /knowledge folder to organize kbs
- [ ] add unit tests
- [ ] add cli support
- [ ] use section drafting style (cli)

