# Russian Literature AI Model Testing

A small blind, rubric-based test of [salakash/SamKash-Tolstoy](https://huggingface.co/salakash/SamKash-Tolstoy), a LoRA adapter on DeepSeek-R1-Distill-Qwen-1.5B. Each prompt was run with the adapter on and off (base model), then scored blind.

- Article: _Medium link to be added after publishing_
- Spreadsheets (easier to view): [Google Drive folder](https://drive.google.com/drive/folders/10_aOcKRGlYg5jrLOqBeaKXe0GGfqSzLz)

## Final run: `run2_temp-0.7/`
**Settings** (from the model's published examples): temperature 0.7, top-p 0.9, 180 new tokens, plain-text prompts (no chat template).

**Prompts** follow the model card's format, `Write a short scene in the style of <work>: <scene>`, with one scene for all three works:
`a woman anxiously waits for a letter from a friend she hasn't seen in a long time.`
Works: *Anna Karenina* (Tolstoy), *Crime and Punishment* (Dostoevsky), *The Lady with the Dog* (Chekhov).

**Design:** 3 works x 2 conditions (adapter / base) x 3 samples = 18 outputs, with the same random seeds for both conditions. Outputs were shuffled within each work, the condition was hidden, and each output was scored by blind ID before the key was opened.

**Six checks per output:** complete prose, on scene, coherent, meta-commentary (counts against), repetition (counts against), literary style (0-2). Score 0-7, shown as a percentage.

| | Adapter | Base |
|---|---|---|
| Overall | 33.3% (21/63) | 33.3% (21/63) |
| Anna Karenina | 28.6% | 19.0% |
| Crime and Punishment | 47.6% | 42.9% |
| The Lady with the Dog | 23.8% | 38.1% |

Checks answered "Yes" (out of 9, adapter vs base): prose 6 vs 3, on scene 6 vs 4, coherent 5 vs 6, meta-commentary 5 vs 1, repetition 9 vs 9. Literary style was 0 for all 18 outputs.

## Earlier run: `run1_temp-0.8/`
An earlier run at temperature 0.8 (more varied outputs), with author names in the prompts and different scenes, plus a stress test outside the model card's scope. It is not used in the article because I reran everything at the model card's settings. The big picture was similar: small differences between adapter and base, and almost no literary style. See the README inside that folder.

## Files
- `run2_temp-0.7/scripts/main.py`: generation script for the final run
- `run2_temp-0.7/`: raw outputs, blind scoring sheet (with notes), blind key, final scoring file
- `run1_temp-0.8/`: files from the earlier run

## Run it
```
pip install -r requirements.txt
python run2_temp-0.7/scripts/main.py
```
The first run downloads the base model and adapter from Hugging Face.

## Caveats
Small sample (9 outputs per condition), one scorer, one scene, one temperature. I settled a few borderline scoring rules while scoring (for example, how to mark outputs that repeat a single sentence) and applied them to all 18 outputs. The model's author suggests lower temperatures (0.4-0.7) for tighter outputs, and I only tested 0.7.
