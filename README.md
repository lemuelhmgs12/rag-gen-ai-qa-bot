# ABC Grocery Help-Desk Assistant (RAG)

A help-desk assistant for a grocery retailer that answers customer questions using **Retrieval Augmented Generation (RAG)**. It retrieves the relevant entries from the company FAQ and answers **only** from them. When the FAQ doesn't cover a question, it hands the customer over to a human instead of guessing.

Built with LangChain, Chroma and OpenAI.

**Full write-up:** _coming soon_

---

## Results

I tested the assistant with 15 questions: 10 the FAQ can answer, worded the way a customer might ask, and 5 it can't.

| Check | Result |
|---|---|
| Retrieval: right FAQ ranked first (in-scope) | **10 / 10** |
| Retrieval: no chunks returned (out-of-scope) | **5 / 5** |
| Assistant: in-scope questions answered | **10 / 10** |
| Assistant: out-of-scope questions refused with the fallback message | **5 / 5** |
| Assistant: answers fully grounded in the FAQ (manual review) | **9 / 10** |
| Assistant: answers with wrong facts | **0 / 10** |

### Key findings

- **The threshold was chosen from real scores.** The weakest correct match scored 0.371 ("Are you hiring?") and the strongest out-of-scope question scored 0.299 ("Do you sell gluten-free bread?"). I set the threshold to **0.33**, in the middle of that gap. The original value of 0.25 let two out-of-scope questions through.
- **The hardest out-of-scope questions sound like shop questions.** "Gluten-free bread" scored more than three times higher than "baby dolphin" (0.090), because it's about food.
- **Refusing works in two layers.** The threshold removes every chunk for out-of-scope questions, and the prompt tells the LLM to give the fallback when the context is empty. The assistant never answered from its own knowledge.
- **One answer added a small detail.** For "Are you hiring?", it said "check our current openings", which suggests there are jobs available. The FAQ doesn't say that.
- **Follow-up questions fail.** The retriever only sees the current message, so "Can you confirm how much *it* costs?" after a question about Delivery Club retrieved nothing, and the assistant gave the fallback. The grounding rules stopped it from saying anything wrong, but the customer didn't get an answer. See [Next steps](#next-steps).

---

## How it works

1. **Load and chunk:** the FAQ (32 Q&A pairs) is split on its `###` headers, giving one chunk per question with the FAQ id kept in the metadata.
2. **Embed and store:** each chunk is embedded with `text-embedding-3-small` and saved in a local Chroma database (cosine similarity).
3. **Retrieve:** for each question, up to `K = 4` chunks scoring at least `0.33` are retrieved.
4. **Answer:** the chunks are passed to `gpt-5` with a system prompt that only allows answers from the retrieved context, and gives a fixed fallback message when the context doesn't have the answer.
5. **Memory:** the chat version keeps the conversation history for each session, which is used only for personalisation (such as the customer's name), never for company facts.

---

## Project structure

```
rag-gen-ai-qa-bot/
├── data/
│   └── abc-grocery-help-desk-data.md   # the help-desk FAQ (32 Q&A pairs)
├── notebooks/
│   ├── 01_load_and_chunk.ipynb         # load the FAQ, split and check the chunks
│   ├── 02_retrieval_testing.ipynb      # test retrieval, choose K and the threshold
│   ├── 03_evaluation.ipynb             # evaluate the full assistant
│   └── 04_conversation_memory.ipynb    # test memory and follow-up questions
├── src/
│   ├── config.py                       # paths, models and retrieval settings
│   ├── ingest.py                       # builds the vector database
│   ├── rag_chain.py                    # retriever + prompt + LLM, with and without memory
│   └── chat.py                         # chat with the assistant in the terminal
├── .env.example
├── pyproject.toml
└── uv.lock
```

---

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. install the dependencies
uv sync

# 2. add your OpenAI API key
cp .env.example .env
# then edit .env and set OPENAI_API_KEY

# 3. build the vector database (run once, or again if the FAQ changes)
uv run python -m src.ingest

# 4. chat with the assistant
uv run python -m src.chat
```

Run all commands from the project root.

---

## Known limitations

- **Small test set:** 15 questions cover common cases, but not everything. 
- **Results come from one run:** LLM answers can be worded differently between runs, even with `temperature=0`.
- **Follow-up questions** that depend on earlier messages don't retrieve the right FAQ (see above).
- **Memory is in-memory only:** the chat history is lost when the program stops.
- **Deprecated memory class:** `ChatMessageHistory` is deprecated in recent LangChain versions (adderss in the next iteration). Versions are pinned in `uv.lock`, so the project works as-is.

---

## Next steps

- **Fix follow-up questions:** rewrite each follow-up into a standalone question using the chat history before retrieving.
- **Move memory to LangGraph's checkpointer**, the replacement LangChain recommends, and store it somewhere persistent such as Redis or a database.
- **Expand the test set**, including questions that need two FAQs and multi-turn conversations.
- **Add a simple web UI**.

---

## Tech stack

Python 3.12 · uv · LangChain · Chroma · OpenAI (`gpt-5`, `text-embedding-3-small`) · pandas · Jupyter