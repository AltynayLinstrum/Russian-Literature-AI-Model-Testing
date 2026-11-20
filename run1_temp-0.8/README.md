# Earlier run (temperature 0.8)

Not used in the article. Kept for anyone who wants to see it.

- **In-scope test:** Dostoevsky, Chekhov, Gogol. Prompt: `Write a short scene in the style of <author>. A man waits at a railway station for a train that is late.` Temperature 0.8, top-p 0.9, 150 new tokens. Adapter 38.1% vs base 34.9%.
- **Stress test:** Tolstoy, Gorky, Pushkin, outside the model card's scope. Scene: a family sits in silence at the dinner table, each staring at a phone. Adapter 19.0% vs base 31.7%. This test was generated with an earlier version of the script that was not saved, so its exact settings were not independently verified.
- Pooled across both tests: adapter 28.6% vs base 33.3%.
- 3 samples per condition per author, same rubric and blind process as the final run.

Folders: `inscope_test/`, `stress_test/`, `scripts/` (move the existing folders here).
