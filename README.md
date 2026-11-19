# Russian Literature AI Model Testing

A small blind, rubric-based test of [salakash/SamKash-Tolstoy](https://huggingface.co/salakash/SamKash-Tolstoy), a LoRA adapter on DeepSeek-R1-Distill-Qwen-1.5B. Each prompt was run with the adapter on and off (base model), then scored blind.

- Article: _Medium link to be added after publishing_
- Spreadsheets (easier to view): [Google Drive folder](https://drive.google.com/drive/folders/10_aOcKRGlYg5jrLOqBeaKXe0GGfqSzLz)

## Method
- Same settings for every output: temperature 0.8, top-p 0.9, first 150 tokens, plain-text prompts (no chat template).
- 3 samples per condition per author, with the same random seeds for adapter and base.
- Outputs shuffled within each author, condition hidden, scored by ID, key kept in a separate file.
- Six checks per output: complete prose, on scene, coherent, meta-commentary, repetition, literary style. Score 0-7, shown as a percentage.

## Tests
| | In-scope test | Stress test |
|---|---|---|
| Authors | Dostoevsky, Chekhov, Gogol | Tolstoy, Gorky, Pushkin |
| Scene | A man waits at a railway station for a train that is late | A family sits in silence at the dinner table after an argument, each one staring at a phone |
| Adapter vs base | 38.1% vs 34.9% | 19.0% vs 31.7% |

## Files
- `scripts/main.py`: generation script. As saved, it reproduces the in-scope test. The stress test used the same setup with a different scene and authors.
- `inscope_test/` and `stress_test/`: raw outputs, blind scoring sheet, blind key, and final scoring file for each test. The stress test generated 5 samples per condition, and the first 3 were scored.
- `early_exploration/` (if present): my first informal try, before the blind test.

## Run it
```
pip install -r requirements.txt
python scripts/main.py
```
The first run downloads the base model and adapter from Hugging Face.

## Caveats
Small sample (9 outputs per condition per test), one scorer, and rubric wording refined during the stress test. See the article for details.
