# Race recap corpus

Twelve original recap documents used to seed the RAG layer (step 3), covering
the 2025 title fight and the first half of the 2026 season (F1's new
power-unit/aero regulations took effect in 2026).

These are **written for this project**, not copied from any news outlet.
Every factual claim (winners, positions, points, grid slots, DNF causes) was
verified against our own `box_box_bot.data.fastf1_client` module — see the
session in project history where each race's classified results were pulled
live and cross-checked before writing. Narrative context (why a result
mattered, driver/team storylines) was researched via web search and then
written in original prose, not quoted.

| File | Season | Round | Race |
|---|---|---|---|
| 2025_r01_australian_gp.md | 2025 | 1 | Australian GP |
| 2025_r04_bahrain_gp.md | 2025 | 4 | Bahrain GP |
| 2025_r16_italian_gp.md | 2025 | 16 | Italian GP |
| 2025_r18_singapore_gp.md | 2025 | 18 | Singapore GP |
| 2025_r23_qatar_gp.md | 2025 | 23 | Qatar GP |
| 2025_r24_abu_dhabi_gp.md | 2025 | 24 | Abu Dhabi GP |
| 2026_r01_australian_gp.md | 2026 | 1 | Australian GP |
| 2026_r02_chinese_gp.md | 2026 | 2 | Chinese GP |
| 2026_r04_miami_gp.md | 2026 | 4 | Miami GP |
| 2026_r08_austrian_gp.md | 2026 | 8 | Austrian GP |
| 2026_r09_british_gp.md | 2026 | 9 | British GP |
| 2026_r12_dutch_gp.md | 2026 | 12 | Dutch GP |

Each file has a YAML frontmatter block (`season`, `round`, `race_name`,
`date`) that the step-3 ingestion script reads into vector store metadata,
so retrieved chunks can cite their source document/race.
