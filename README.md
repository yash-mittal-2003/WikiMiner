# WikiMiner README
*A single‑pass Wikipedia crawler for building parallel sentence corpora across 11 Indic languages*  

---

## Repository Layout

| Path | Purpose |
|------|---------|
| **`WikiMiner_final.ipynb`** | Crawls Wikipedia, builds 15 threshold‑indexed parallel corpora (11 Indic languages), projects NER tags, and writes statistics. |
| **`Fine_tune_comparison.ipynb`** | Compares two Indic‑BERT fine‑tunes: (i) gold **Naamapadam** only, (ii) gold + Wikipedia silver. |

---

## 1  What the script does & why
| Stage | Goal | Core idea |
|-------|------|-----------|
| **1. Seed selection** | Start from carefully chosen *English* categories that are likely to have Indic counterparts (e.g. “Hindi cinema”, “Kannada literature”). | Gives topic coherence and avoids crawling the whole of Wikipedia. |
| **2. Article pairing** | For each English article, follow the **langlink** to its version in the target Indic language (if one exists). | Guarantees the two texts are about the same entity/event. |
| **3. Sentence‑level alignment** | • Split both pages into sentences.<br>• Embed sentences with **LaBSE** (multilingual BERT).<br>• Compute cosine‑similarity matrix.<br>• Keep mutual best matches (optional length filter + SimAlign word‑coverage). | Produces candidate parallel pairs without any trained MT model. |
| **4. Threshold grid filtering** | Instead of re‑crawling 15 times, we **capture all pairs once** (τ = 0, α = 0) and post‑filter in‑memory for every (τ, α) combination. | Saves network/compute time, lets you explore quality‑vs‑quantity trade‑offs offline. |
| **5. Corpus & stats export** | Write JSONL files under `parallel_datasets/<τ>/<α>/` and matching `stats/<τ>/<α>/`. | Easy to load in any framework; statistics help pick the best threshold. |

---

## 2  Key implementation nuances

### 2.1 Single HTTP session with retries
```python
session = requests.Session()
session.headers.update({
    "User-Agent": "WikiMiner/1.0 (your_email@example.com)"
})
session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=1.5,
        status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])))
```
* **Why?** Wikimedia throttles anonymous bursts with **HTTP 429**.  
  A custom User‑Agent + exponential back‑off satisfies their API etiquette and eliminates the error.

### 2.2 Avoiding duplicate API calls
We cache the langlink during the first lookup; every article therefore triggers exactly **one** `prop=langlinks` request and two page fetches.

### 2.3 Sentence alignment details
* **LaBSE** `pooler_output` → language‑agnostic 768‑d vectors.  
* **Mutual‑best strategy** filters out many false friends.  
* **SimAlign** (word‑level BERT alignment) can be re‑enabled by raising `alignment_threshold`.

### 2.4 Parameter grid
```python
SIM_THRESHOLDS   = [0.6, 0.7, 0.8]      # τ (cosine)
ALIGN_THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]  # α (word‑coverage)
```
Total of **3×5 = 15** corpora per language produced in one run.

---

## 3  Statistics: what they measure

| Field | Level | Meaning |
|-------|-------|---------|
| `num_pairs` | aggregate & category | raw size of the corpus after filters. |
| `mean_cosine` | "" | average semantic similarity → proxy for translation quality. |
| `mean_alignment` | "" | average word‑coverage from SimAlign → structural faithfulness. |
| `avg_en_len`, `avg_tgt_len`, `avg_len_ratio`| "" |	sentence‑length sanity checks. |
| `avg_token_coverage` | "" | fraction of English tokens aligned (how many tags survive projection) |

Per‑category roll‑ups highlight topical domains that align well (e.g. literature) versus those that struggle (e.g. song lyrics).

---

## 4  Directory layout of outputs

```
parallel_datasets/
└── 0.7/                # τ
    └── 0.6/            # α
        └── parallel_dataset_hindi.json
stats/
└── 0.7/
    └── 0.6/
        └── stats_hindi.json
```

Each JSONL corpus line looks like:
```json
{"English": "Tagore was a polymath.", "Hindi": "टैगोर एक बहुमुखी प्रतिभा थे।"}
```

---

## 5  How to run

### 5.1 Inside a notebook (recommended)
```python
# crawl just Assamese with τ=0.7, α=0.6
main(langs=['as'], sim_thresholds=[0.7], alpha_thresholds=[0.6])

# Bengali + Gujarati with default grids
main(langs=['bn', 'gu'])

# All languages but a custom τ grid
main(sim_thresholds=[0.55, 0.65, 0.75])
```

### 5.2 From the shell
If you save the script as `wiki_miner.py` you can still launch the default run:
```bash
python wiki_miner.py          # all languages, full grids
```
(Flags can be added via `argparse` if needed, but the notebook API already covers flexible experimentation.)

---

## 6  Interpreting threshold effects

* **Raise τ / α** → fewer pairs, higher `mean_cosine`, higher precision.  
* **Lower τ / α** → more pairs, noisier; good for back‑translation or large‑scale pre‑training.  
Plotting `num_pairs` vs. `mean_cosine` across the 15 cells gives an immediate quality‑volume Pareto frontier.

---

## 7  Common pitfalls & remedies

| Symptom | Cause | Fix |
|---------|-------|-----|
| **HTTP 429** even with session | Running multiple instances from one IP | Increase `backoff_factor`, or stagger runs. |
| Memory blow‑up | Very large language set in one session | Run `main()` per language in separate processes or write `all_pairs` to disk incrementally. |
| SpaCy sentence splitter missing | Model not installed | `python -m spacy download en_core_web_sm`. |

---

## 8  Dependencies
```bash
pip install sentence-transformers torch spacy tqdm pandas simalign \
            scikit-learn requests beautifulsoup4 indic-nlp-library
python -m spacy download en_core_web_sm
```
Don’t forget to clone `indic_nlp_resources` and set `INDIC_RESOURCES_PATH`.

---

## 9  Fine‑Tune Comparison Notebook (`Fine_tune_comparison.ipynb`)

| Step&nbsp;# | Action |
|-------------|--------|
| **1 Data selection** | Pick a `(τ, α)` slice (default: Assamese, τ = 0.6, α = 0.4). |
| **2 Dataset merge** | Combine gold **Naamapadam** with (optionally) the Wikipedia silver slice. |
| **3 Token alignment** | Indic‑BERT tokenizer aligns word‑pieces → BIO tags (sub‑word indices masked with `‑100`). |
| **4 Two runs** | **Baseline** = gold only · **Augmented** = gold + silver. |
| **5 Logging** | Epoch‑wise loss / precision / recall / F1 printed; learning‑curve PNG written to `figures/`. |
| **6 Results** | Per‑epoch metrics exported as **CSV** and **XLSX** under `results/`. |

```python
# Example cell – run inside Fine_tune_comparison.ipynb
run_experiment(
    lang        = "as",   # Assamese
    tau         = 0.6,
    alpha       = 0.4,
    num_epochs  = 10,
    batch_size  = 32
)
```

### Enjoy mining parallel corpora!  
For any questions or improvements, feel free to reach out.
