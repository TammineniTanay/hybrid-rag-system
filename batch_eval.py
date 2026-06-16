import requests, json, time, sys

# Configuration
API_URL = "http://localhost:8000/api/query"
OUTPUT_FILE = "eval_results_50.json"
TIMEOUT = 300
MAX_RETRIES = 3
RETRY_DELAY = 2

questions = [
    # === Attention Is All You Need (10 questions) ===
    "What is the transformer architecture?",
    "How does scaled dot-product attention work in the Transformer?",
    "What is multi-head attention and how many heads does the Transformer use?",
    "What BLEU score did the Transformer achieve on WMT 2014 English-to-German?",
    "How does the Transformer use positional encoding?",
    "What regularization techniques does the Transformer use during training?",
    "How does self-attention complexity compare to recurrent layer complexity?",
    "What is the feed-forward network structure in each Transformer layer?",
    "How long did it take to train the big Transformer model?",
    "What is the encoder-decoder structure of the Transformer?",

    # === BERT (10 questions) ===
    "What is BERT and what does it stand for?",
    "How does BERT's masked language model pre-training work?",
    "What is the next sentence prediction task in BERT?",
    "What are the differences between BERT-BASE and BERT-LARGE?",
    "How does BERT differ from OpenAI GPT in its use of attention?",
    "What results did BERT achieve on the SQuAD benchmark?",
    "What pre-training data was used to train BERT?",
    "How does BERT handle input representation with segment and position embeddings?",
    "What is the effect of model size on BERT's fine-tuning performance?",
    "How does BERT's feature-based approach compare to fine-tuning on NER?",

    # === RAG (10 questions) ===
    "What is retrieval-augmented generation and how does RAG work?",
    "What is the difference between RAG-Sequence and RAG-Token models?",
    "What retriever does RAG use and how does it work?",
    "What generator model does RAG use as its parametric memory?",
    "How does RAG perform on open-domain question answering tasks?",
    "What is the index hot-swapping capability of RAG?",
    "How does RAG handle fact verification on the FEVER dataset?",
    "What are the generation diversity results for RAG compared to BART?",
    "How does the number of retrieved documents affect RAG performance?",
    "What are the advantages of RAG over purely parametric models like T5?",

    # === Self-RAG (10 questions) ===
    "What is Self-RAG and how does it differ from standard RAG?",
    "What are reflection tokens in Self-RAG?",
    "How does Self-RAG decide when to retrieve passages?",
    "What are the ISREL, ISSUP, and ISUSE critique tokens in Self-RAG?",
    "How was the Self-RAG critic model trained?",
    "What datasets was Self-RAG evaluated on?",
    "How does Self-RAG perform compared to ChatGPT on open-domain QA?",
    "How does Self-RAG enable test-time customization of model behavior?",
    "What is the tree-decoding inference strategy in Self-RAG?",
    "What ablation results show the importance of Self-RAG's components?",

    # === CRAG (10 questions) ===
    "What is Corrective Retrieval Augmented Generation and why is it needed?",
    "How does CRAG's retrieval evaluator assess document quality?",
    "What are the three action triggers in CRAG: Correct, Incorrect, and Ambiguous?",
    "How does CRAG's knowledge refinement process work?",
    "How does CRAG use web search as a knowledge correction strategy?",
    "How does CRAG compare to Self-RAG on the PopQA dataset?",
    "What is the decompose-then-recompose algorithm in CRAG?",
    "How robust is CRAG when retrieval quality degrades?",
    "What is the computational overhead of CRAG compared to standard RAG?",
    "How does CRAG's T5-based retrieval evaluator compare to ChatGPT for evaluating retrieval?",
]

def query_with_retry(question, index):
    """
    Query the RAG API with automatic retry on failure.
    Retries up to MAX_RETRIES times with exponential backoff.
    """
    for attempt in range(MAX_RETRIES):
        start = time.time()
        try:
            resp = requests.post(
                API_URL,
                json={"question": question},
                timeout=TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            data["latency_s"] = round(time.time() - start, 2)
            data["attempts"] = attempt + 1
            return data
        except requests.exceptions.Timeout:
            elapsed = round(time.time() - start, 2)
            print(f"  Attempt {attempt+1}/{MAX_RETRIES} timed out after {elapsed}s")
        except requests.exceptions.ConnectionError:
            print(f"  Attempt {attempt+1}/{MAX_RETRIES} connection error — is the server running?")
        except Exception as e:
            print(f"  Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))

    # All retries exhausted
    return {
        "question": question,
        "error": f"Failed after {MAX_RETRIES} attempts",
        "latency_s": 0,
        "attempts": MAX_RETRIES
    }

def run_evaluation(questions, output_file):
    """
    Run the full batch evaluation and save results.
    Prints progress and summary statistics on completion.
    """
    results = []
    errors = 0
    total_start = time.time()

    print(f"Starting evaluation of {len(questions)} questions...")
    print(f"API: {API_URL}")
    print("-" * 60)

    for i, question in enumerate(questions):
        data = query_with_retry(question, i)
        results.append(data)

        status = "ERROR" if "error" in data else "OK"
        if status == "ERROR":
            errors += 1

        print(f"[{i+1}/{len(questions)}] {status} {data.get('latency_s', 0):.1f}s - {question[:60]}")

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    total_time = round(time.time() - total_start, 1)
    successful = len(questions) - errors
    avg_latency = sum(r.get("latency_s", 0) for r in results) / len(results)

    print("\n" + "=" * 60)
    print(f"Evaluation Complete")
    print(f"  Total questions : {len(questions)}")
    print(f"  Successful      : {successful}")
    print(f"  Errors          : {errors}")
    print(f"  Avg latency     : {avg_latency:.2f}s")
    print(f"  Total time      : {total_time}s")
    print(f"  Results saved   : {output_file}")
    print("=" * 60)

    return results

if __name__ == "__main__":
    run_evaluation(questions, OUTPUT_FILE)