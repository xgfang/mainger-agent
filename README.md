# mainger-agent (online)

LLM-driven assistant for the `mainger` framework. Bring your own API key
(Anthropic, OpenAI, or Google), keep your data on your machine, and let
the agent walk through detecting the regime, computing the theoretical
$\eta$ bound, running the concordance diagnostic, and producing three
artifacts: an integration **report**, runnable R **code**, and a plain-
language **explanation**.

## File map

```
mainger-agent-online/
├── README.md              ← this file
├── requirements.txt       ← Python deps
├── .env.example           ← copy to .env, fill in your API key
├── config.yaml            ← which vendor + which model to use
├── agent.py               ← orchestration loop (CLI entry)
├── llm_client.py          ← vendor-neutral wrapper (Anthropic/OpenAI)
├── tools.py               ← tool specs + Python→Rscript dispatch
├── data_io.py             ← CSV/parquet/manual-entry parsers
├── skill.md               ← system prompt + framework reference
├── templates/
│   ├── report.md.j2       ← integration report template
│   ├── code.R.j2          ← runnable R code template
│   └── explanation.md.j2  ← plain-language explanation template
├── r_helpers/
│   └── run_mainger.R      ← R bridge — calls the mainger package
└── examples/
    └── partial_example.csv
```

## What you need to fill in

Five things, all marked with `# FILL IN:` comments in the code:

1. **`.env`** — your API key (`ANTHROPIC_API_KEY=...` or `OPENAI_API_KEY=...`)
2. **`config.yaml`** — pick `vendor` (anthropic / openai) and `model` (e.g. `claude-opus-4-7`, `gpt-4o`)
3. **`r_helpers/run_mainger.R`** — set the path to your installed `mainger` package
   (only needed if it's not on the default library path)
4. **`skill.md`** — already filled in, but you may want to tune the tone /
   add domain-specific guidance (eGFR, PRS, etc.)
5. **`templates/*.j2`** — adjust the report sections / code style to match
   what your users expect

## Install and run

```bash
# clone, then:
pip install -r requirements.txt
cp .env.example .env          # then edit .env to add your API key
Rscript -e 'install.packages("mainger_0.2.0.tar.gz", repos=NULL, type="source")'

# run on the included example
python agent.py --input examples/partial_example.csv \
                --external-coef examples/external_coef.csv \
                --regime partial \
                --out-dir runs/demo
```

Outputs go to `runs/demo/`:
- `report.md` — the integration report
- `analysis.R` — the runnable R script
- `explanation.md` — the plain-language summary
- `trace.json` — every tool call the LLM made (for auditability)

## Multi-vendor note

`llm_client.py` wraps Anthropic and OpenAI behind one interface. Adding
Google Gemini is a ~30-line addition once you have a `google-genai`
client; the tool-call format is similar enough that the same tool specs
work after one schema translation. Mention this in the paper as
"vendor-neutral via standard function-calling APIs" rather than promising
all three on day one.
