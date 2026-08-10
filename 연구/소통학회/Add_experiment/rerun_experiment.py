# -*- coding: utf-8 -*-
"""
Re-run harness for the two robustness conditions the 2026-05-20 data cannot answer.

    export OPENROUTER_API_KEY=...          # or put it in a .env beside this file
    python rerun_experiment.py --plan      # cost/call estimate, queries nothing
    python rerun_experiment.py --run

WHAT THIS ADDS OVER THE ORIGINAL SCRIPTS
  1. Repeated trials.  Each item is asked R times so test-retest stability can be
     estimated. The original design asked once, which is the manuscript's first-listed
     limitation.
  2. Anchor-reversed arm.  Dominguez-Olmedo et al. (2024) show models are sensitive to
     the ORDER OF RESPONSE OPTIONS. This arm presents every scale with its anchors
     reversed and maps the returned integer back (x -> max+min-x), so a position-anchored
     model and a content-anchored model give different results.

     Note this is NOT about the sequence in which items were presented. Every item is an
     independent single-turn request with no conversation history, so presentation order
     has no causal path to the response. That critique is answerable by argument; option
     order is not, and needs this arm.
  3. Two fixes to the original instrumentation, both of which change what gets recorded
     rather than what gets asked:
       - Study 1 parsed answers with int(''.join(filter(str.isdigit, text))), which
         silently concatenates every digit in the reply: "8/10" became 810. Here the reply
         is parsed to a candidate integer and REJECTED if it falls outside the item's own
         scale, so an unusable reply is recorded as unusable instead of as data.
       - The raw reply text, finish_reason, latency and the model id OpenRouter actually
         served are all stored. The last one matters: the identifiers are floating
         aliases, and without the resolved id a future reader cannot tell which weights
         answered.

  Both arms must be collected in the SAME session. Differencing a run today against
  2026-05-20 would confound the effect of interest with three months of alias drift.

Output: results/<run_id>/responses.jsonl  - one JSON object per API call, appended as it
completes. Re-running with the same --run-id skips calls already present, so an
interrupted run resumes where it stopped.
"""

import argparse, csv, itertools, json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    sys.exit("requests is required:  pip install requests")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    "openai/gpt-5.5",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-4.3",
    "meta-llama/llama-4-maverick",
    "mistralai/mistral-medium-3-5",
    "deepseek/deepseek-v4-pro",
]

SYSTEM_S1 = ("You must answer the following question. Return ONLY a single integer "
             "corresponding to your choice. Do not include any explanations, "
             "introductory text, or punctuation.")
SYSTEM_S2 = ("You are participating in a risk perception survey. Read the question and "
             "the scale carefully. You MUST return ONLY a single integer between 1 and 5. "
             "Do not include any words, punctuation, explanations, or periods.")

RISK_ISSUES = ["Sunbathing", "Food colourings", "Genetic engineering", "Nuclear power",
               "Mugging", "Home accidents", "Ozone depletion", "Car driving",
               "Microwave ovens", "AIDS", "War", "Terrorism", "Alcoholic drinks"]

OUTRAGE_FACTORS = {
    "Involuntariness": ("Do you think people are exposed to {risk_issue} voluntarily or "
                        "involuntarily?", "Risk assumed voluntarily", "Risk assumed involuntarily"),
    "Delayed_effects": ("Do you think the harmful effects of {risk_issue} are likely to occur "
                        "at some later time?", "No effect delayed", "Many effects delayed"),
    "Severity": ("Do you think people will die from {risk_issue} when things go wrong?",
                 "Certain not to be fatal", "Certain to be fatal"),
    "Dread": ("Are the risks of {risk_issue} ones that you can think about reasonably calmly, "
              "or are they ones that you have a great dread for on the level of gut reaction?",
              "Think about calmly", "Think about with dread"),
    "Catastrophic_potential": ("Do you think {risk_issue} can harm a large number of people at "
                               "once or only one person at a time?",
                               "One person at a time", "Many people at once"),
    "Harm_to_future_generations": ("Do you think {risk_issue} poses risks to future generations?",
                                   "Very little threat", "Very great threat"),
    "Lack_of_knowledge_exposed": ("Do you think the risks associated with {risk_issue} are well "
                                  "known by the persons who are exposed to the risk?",
                                  "Risk well known", "Risk not known"),
    "Lack_of_knowledge_scientists": ("Do you think the risks associated with {risk_issue} are "
                                     "well known by scientists?", "Risk well known", "Risk not known"),
    "Unfairness": ("Do you think that the people who are exposed to the risks of {risk_issue} are "
                   "the same as those who receive the benefits?",
                   "Same people / Fair", "Not same people / Unfair"),
}
OVERALL = {
    "Riskiness": ("On a scale of 1 to 5, how much risk do you think is associated with "
                  "{risk_issue}?", "Very low risk", "Very high risk"),
    "Unacceptability": ("On a scale of 1 to 5, how acceptable do you feel the current risk is "
                        "for {risk_issue}?", "Completely acceptable", "Completely unacceptable"),
}


def load_json(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)["questions"]


def build_items(arm):
    """Every (study, item_id, prompt, scale_min, scale_max, reversed?) to be asked."""
    items = []

    # ---- Study 1: WVS. Anchor reversal needs authored variants; the option text is
    # embedded in each question, so it cannot be flipped by string surgery.
    suffix = "_reversed" if arm == "optrev" else ""
    for axis, fname in (("Grid", "grid_questions"), ("Group", "group_questions")):
        path = os.path.join(HERE, f"{fname}{suffix}.json")
        if not os.path.exists(path):
            if arm == "optrev":
                print(f"  ! {fname}{suffix}.json missing - Study 1 skipped for arm 'optrev'")
                continue
            sys.exit(f"missing required file: {path}")
        for i, q in enumerate(load_json(f"{fname}{suffix}.json"), 1):
            items.append({"study": "study1", "item_id": f"{axis}_Q{i}",
                          "prompt": q["question"], "scale_min": 1, "scale_max": q["scale"],
                          "system": SYSTEM_S1, "issue": "",
                          "anchors_reversed": arm == "optrev"})

    # ---- Study 2: the scale line is generated, so reversal is exact and automatic.
    for issue in RISK_ISSUES:
        for name, (stem, lo, hi) in {**OUTRAGE_FACTORS, **OVERALL}.items():
            a, b = (hi, lo) if arm == "optrev" else (lo, hi)
            prompt = f"{stem.format(risk_issue=issue)}\nScale: 1 ({a}) to 5 ({b})"
            items.append({"study": "study2", "item_id": name, "prompt": prompt,
                          "scale_min": 1, "scale_max": 5, "system": SYSTEM_S2,
                          "issue": issue, "anchors_reversed": arm == "optrev"})
    return items


def parse_answer(text, lo, hi):
    """Return (value, status). Never invents a value it cannot justify.

    The original Study 1 parser concatenated every digit in the reply; this one takes
    the first standalone integer and rejects it if the item's own scale cannot contain
    it. An out-of-range reply is a parse failure, not an observation.
    """
    if text is None:
        return None, "empty_response"
    m = re.search(r"-?\d+", text)
    if not m:
        return None, "no_integer_in_reply"
    v = int(m.group())
    if not (lo <= v <= hi):
        return None, f"out_of_scale_range_{v}"
    return v, "ok"


def call(api_key, model, item, rep, arm, timeout, max_retries=3):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "HTTP-Referer": "http://localhost:8000",
               "X-Title": "Cultural Prototype Research (robustness re-run)"}
    payload = {"model": model,
               "messages": [{"role": "system", "content": item["system"]},
                            {"role": "user", "content": item["prompt"]}],
               "temperature": 0.0, "top_p": 1.0, "max_tokens": 1000}
    rec = {"arm": arm, "rep": rep, "model_requested": model, "study": item["study"],
           "issue": item["issue"], "item_id": item["item_id"],
           "anchors_reversed": item["anchors_reversed"],
           "scale_min": item["scale_min"], "scale_max": item["scale_max"],
           "prompt": item["prompt"]}
    last = ""
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            r = requests.post(API_URL, headers=headers, data=json.dumps(payload),
                              timeout=timeout)
            rec["latency_s"] = round(time.time() - t0, 2)
            if r.status_code != 200:
                last = f"http_{r.status_code}: {r.text[:200]}"
                if r.status_code in (400, 401, 403, 404):
                    break                      # not worth retrying
                time.sleep(2 * attempt); continue
            body = r.json()
            choice = body["choices"][0]
            text = (choice.get("message") or {}).get("content")
            value, status = parse_answer(text, item["scale_min"], item["scale_max"])
            rec.update({
                "raw_response": text, "finish_reason": choice.get("finish_reason"),
                "model_served": body.get("model"), "generation_id": body.get("id"),
                "usage": body.get("usage"), "attempts": attempt,
                "value_as_presented": value, "parse_status": status,
                # comparable with the 2026-05-20 data regardless of arm
                "value_normalised_orientation":
                    None if value is None else
                    (item["scale_min"] + item["scale_max"] - value
                     if item["anchors_reversed"] else value),
            })
            return rec
        except Exception as e:
            rec["latency_s"] = round(time.time() - t0, 2)
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 * attempt)
    rec.update({"raw_response": None, "value_as_presented": None,
                "value_normalised_orientation": None, "parse_status": "failed",
                "error": last, "attempts": max_retries})
    return rec


def key_of(rec):
    return (rec["arm"], rec["rep"], rec["model_requested"], rec["study"],
            rec["issue"], rec["item_id"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="rerun-01")
    ap.add_argument("--arms", default="baseline,optrev",
                    help="baseline = exact replication; optrev = anchors reversed")
    ap.add_argument("--reps-study1", type=int, default=5)
    ap.add_argument("--reps-study2", type=int, default=3)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    jobs = []
    for arm in arms:
        items = build_items(arm)
        for item in items:
            reps = args.reps_study1 if item["study"] == "study1" else args.reps_study2
            for rep in range(1, reps + 1):
                for m in models:
                    jobs.append((m, item, rep, arm))

    print(f"run-id      {args.run_id}")
    print(f"arms        {', '.join(arms)}")
    print(f"models      {len(models)}")
    print(f"reps        study1 x{args.reps_study1}   study2 x{args.reps_study2}")
    for arm in arms:
        for st in ("study1", "study2"):
            n = sum(1 for m, i, r, a in jobs if a == arm and i["study"] == st)
            print(f"  {arm:<9} {st}  {n:>6} calls")
    print(f"TOTAL       {len(jobs):,} calls")
    print("\nEach call sends ~120 input tokens and asks for one integer back. Cost is")
    print("dominated by per-call overhead on the reasoning-tier models; check current")
    print("OpenRouter pricing before a full run. Use --reps-study2 1 for a smoke test.")

    outdir = os.path.join(HERE, "..", "results", args.run_id)
    outfile = os.path.join(outdir, "responses.jsonl")
    done = set()
    if os.path.exists(outfile):
        with open(outfile, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(key_of(json.loads(line)))
                except Exception:
                    pass
        print(f"\nresuming: {len(done):,} calls already recorded, "
              f"{len(jobs) - len(done):,} remaining")

    if args.plan or not args.run:
        print("\n(planning only - nothing was queried. add --run to execute)")
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        envf = os.path.join(HERE, ".env")
        if os.path.exists(envf):
            for line in open(envf, encoding="utf-8"):
                if line.strip().startswith("OPENROUTER_API_KEY"):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set (environment or .env beside this script)")

    os.makedirs(outdir, exist_ok=True)
    pending = [j for j in jobs
               if (j[3], j[2], j[0], j[1]["study"], j[1]["issue"], j[1]["item_id"]) not in done]
    random.Random(20260809).shuffle(pending)   # spread load across providers

    t0, n_ok, n_bad = time.time(), 0, 0
    with open(outfile, "a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call, api_key, m, i, r, a, args.timeout): (m, i, r, a)
                   for m, i, r, a in pending}
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if rec["parse_status"] == "ok":
                n_ok += 1
            else:
                n_bad += 1
            if n % 25 == 0 or n == len(pending):
                el = time.time() - t0
                print(f"  {n:>6}/{len(pending)}  ok {n_ok}  unusable {n_bad}  "
                      f"{el/60:.1f} min  eta {(el/n)*(len(pending)-n)/60:.1f} min")

    print(f"\ndone. {n_ok} usable, {n_bad} unusable -> {outfile}")
    print("next:  python analyse_rerun.py --run-id " + args.run_id)


if __name__ == "__main__":
    main()
