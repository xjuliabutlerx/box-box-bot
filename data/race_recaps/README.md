# Race recap corpus

Thirty-six original recap documents used to seed the RAG layer (step 3): sixteen genuinely historical races spanning 2016-2024 (only the ones that actually mattered for a season's story - title deciders, breakthrough wins, defining incidents, not full-season coverage), plus the 2025 title fight and the full first three quarters of the 2026 season (F1's new power-unit/aero regulations took effect in 2026).

These are **written for this project**, not copied from any news outlet. Every factual claim (winners, positions, points, grid slots, DNF causes) was verified against our own `box_box_bot.data.fastf1_client` module — see the session in project history where each race's classified results were pulled live and cross-checked before writing. Narrative context (why a result mattered, driver/team storylines) was researched via web search and then written in original prose, not quoted.

The eight 2026 rounds added later (3, 5, 6, 7, 10, 11, 13, 14) followed the same fastf1-verification discipline, but for narrative context relied on cross-referencing the *other* recaps already in this corpus rather than web search — several already-written rounds explicitly forward- or back-reference these gaps (e.g. round 4's recap already named rounds 2 and 3 as Antonelli's first two wins; round 9's named round 10 as where Red Bull would revert its rear wing design), so those existing claims constrained and cross-checked what the new recaps had to say to stay consistent with the rest of the corpus.

The sixteen 2016-2024 recaps cover *real* Formula 1 history rather than this project's synthetic 2025/2026 seasons - confirmed live that `fastf1_client` returns genuine historical results for these seasons (e.g. Verstappen's real 2016 Spain debut win, Hamilton/Rosberg's real title fight), not the same kind of synthetic data seeded for the current season. Selecting *which* races "mattered" was an editorial judgment call, not an exhaustive pull of every round - each pick is a title decider, a breakthrough result, or a defining on-track incident, cross-checked against fastf1's actual results/grid positions/DNF causes for every specific factual claim before writing (this caught and fixed two real drafting errors: an overstated Vettel result in the 2017 Mexican GP recap, and an incorrect "finished outside the points" claim for Russell in the 2020 Sakhir GP recap - both wrong until checked against the actual data).

| File | Season | Round | Race |
|---|---|---|---|
| 2016_r05_spanish_gp.md | 2016 | 5 | Spanish GP |
| 2016_r21_abu_dhabi_gp.md | 2016 | 21 | Abu Dhabi GP |
| 2017_r14_singapore_gp.md | 2017 | 14 | Singapore GP |
| 2017_r18_mexican_gp.md | 2017 | 18 | Mexican GP |
| 2018_r11_german_gp.md | 2018 | 11 | German GP |
| 2018_r18_united_states_gp.md | 2018 | 18 | United States GP |
| 2019_r13_belgian_gp.md | 2019 | 13 | Belgian GP |
| 2019_r14_italian_gp.md | 2019 | 14 | Italian GP |
| 2020_r15_bahrain_gp.md | 2020 | 15 | Bahrain GP |
| 2020_r16_sakhir_gp.md | 2020 | 16 | Sakhir GP |
| 2021_r10_british_gp.md | 2021 | 10 | British GP |
| 2021_r22_abu_dhabi_gp.md | 2021 | 22 | Abu Dhabi GP |
| 2022_r18_japanese_gp.md | 2022 | 18 | Japanese GP |
| 2023_r15_singapore_gp.md | 2023 | 15 | Singapore GP |
| 2024_r06_miami_gp.md | 2024 | 6 | Miami GP |
| 2024_r24_abu_dhabi_gp.md | 2024 | 24 | Abu Dhabi GP |
| 2025_r01_australian_gp.md | 2025 | 1 | Australian GP |
| 2025_r04_bahrain_gp.md | 2025 | 4 | Bahrain GP |
| 2025_r16_italian_gp.md | 2025 | 16 | Italian GP |
| 2025_r18_singapore_gp.md | 2025 | 18 | Singapore GP |
| 2025_r23_qatar_gp.md | 2025 | 23 | Qatar GP |
| 2025_r24_abu_dhabi_gp.md | 2025 | 24 | Abu Dhabi GP |
| 2026_r01_australian_gp.md | 2026 | 1 | Australian GP |
| 2026_r02_chinese_gp.md | 2026 | 2 | Chinese GP |
| 2026_r03_japanese_gp.md | 2026 | 3 | Japanese GP |
| 2026_r04_miami_gp.md | 2026 | 4 | Miami GP |
| 2026_r05_canadian_gp.md | 2026 | 5 | Canadian GP |
| 2026_r06_monaco_gp.md | 2026 | 6 | Monaco GP |
| 2026_r07_barcelona_gp.md | 2026 | 7 | Barcelona GP |
| 2026_r08_austrian_gp.md | 2026 | 8 | Austrian GP |
| 2026_r09_british_gp.md | 2026 | 9 | British GP |
| 2026_r10_belgian_gp.md | 2026 | 10 | Belgian GP |
| 2026_r11_hungarian_gp.md | 2026 | 11 | Hungarian GP |
| 2026_r12_dutch_gp.md | 2026 | 12 | Dutch GP |
| 2026_r13_italian_gp.md | 2026 | 13 | Italian GP |
| 2026_r14_spanish_gp.md | 2026 | 14 | Spanish GP (Madrid) |

Each file has a YAML frontmatter block (`season`, `round`, `race_name`, `date`) that the step-3 ingestion script reads into vector store metadata, so retrieved chunks can cite their source document/race.
