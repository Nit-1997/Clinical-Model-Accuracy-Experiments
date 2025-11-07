#!/usr/bin/env python3

"""
inference_vllm.py

Simple concurrent vLLM Chat Completions runner.
- Uses the vLLM OpenAI-like endpoint: POST http://host:port/v1/chat/completions
- Sends batches of concurrent requests (default batch size 50).
- Optionally read only the first N rows via --limit (useful for quick tests).
- Writes output JSONL lines: {"id": "<id>", "pred": <parsed_json_or_empty_dict>}
- Prints total execution time.

Example:
  export VLLM_ENDPOINT=http://127.0.0.1:8000/v1
  export OPENAI_API_KEY=dummy
  python inference_vllm.py --model google/gemma-3-270m-it --out gemma-3-270m-it_outputs.jsonl --limit 10

"""
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests


def extract_json(text: str) -> dict:
    if not text:
        return {}
    # strip code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return {}
    cand = text[s:e + 1]
    cand = re.sub(r",\s*}", "}", cand)
    cand = re.sub(r",\s*]", "]", cand)
    try:
        return json.loads(cand)
    except Exception:
        cand2 = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", cand)
        try:
            return json.loads(cand2)
        except Exception:
            return {}


def call_once(endpoint, headers, model, prompt_text, max_tokens, temperature, timeout):
    payload = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You are a medical electronic records expert. Return ONLY a compact JSON object per the rules."},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"}
    }
    r = requests.post(endpoint.rstrip("/") + "/chat/completions",
                      json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return msg.get("content", "") if isinstance(msg, dict) else str(msg)


def post_completion(endpoint, api_key, model, prompt_text, max_tokens, temperature, timeout):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # first try
    txt = call_once(endpoint, headers, model, prompt_text, max_tokens, temperature, timeout)
    pred = extract_json(txt)
    if pred:  # good
        return pred
    # one strict retry
    retry_prompt = prompt_text + "\n\nReturn ONLY a valid JSON object with the required keys. No prose."
    txt = call_once(endpoint, headers, model, retry_prompt, max_tokens, temperature, timeout)
    print(txt)
    return extract_json(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--notes", default="synthetic_notes.jsonl")
    ap.add_argument("--prompt-template", default="prompt_template.txt",
                    help="The strict prompt with rules & example (will inject NOTE_TEXT).")
    ap.add_argument("--endpoint", default=os.environ.get("VLLM_ENDPOINT",
                                                         os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    ap.add_argument("--limit", type=int, default=10)  # <- default to 10 as you asked
    ap.add_argument("--batch-size", type=int, default=50)  # <- 50 concurrent requests
    ap.add_argument("--max-tokens", type=int, default=64)  # small JSON, keep tight
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    endpoint = args.endpoint
    notes = [json.loads(l) for l in Path(args.notes).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit > 0:
        notes = notes[:args.limit]

    total = len(notes)
    if total == 0:
        print("[ERR] no items to process", file=sys.stderr);
        sys.exit(1)

    prompt_t = Path(args.prompt_template).read_text(encoding="utf-8")

    def build_prompt(note_text: str) -> str:
        # Use the strict template with rules & example; inject the note
        return prompt_t.replace("{{NOTE_TEXT}}", note_text)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as fout:
        for i in range(0, total, args.batch_size):
            batch = notes[i:i + args.batch_size]
            with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                futs = {}
                for it in batch:
                    nid = it.get("id")
                    ptxt = build_prompt(it.get("note", ""))
                    fut = ex.submit(post_completion, endpoint, args.api_key, args.model,
                                    ptxt, args.max_tokens, args.temperature, args.timeout)
                    futs[fut] = nid
                for fut in as_completed(futs):
                    nid = futs[fut]
                    try:
                        pred = fut.result()
                    except Exception as e:
                        print(f"[ERR] id={nid}: {e}", file=sys.stderr)
                        pred = {}
                    fout.write(json.dumps({"id": nid, "pred": pred}, ensure_ascii=False) + "\n")
            print(f"[INFO] processed {min(i + args.batch_size, total)}/{total}", file=sys.stderr)

    print(f"[DONE] {total} items in {time.perf_counter() - t0:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()