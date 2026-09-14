# Track/circuit info corpus

Twenty-three original circuit profile documents used to seed a second RAG collection (separate from `data/race_recaps/`), covering every circuit on the 2026 season calendar. These back `strategist_agent`'s `search_track_info` tool, used for *qualitative* circuit character - overtaking difficulty, tire degradation tendencies, elevation, DRS zones, typical one-stop-vs-two-stop pattern - none of which the structured fastf1 tools capture (they cover session/race-specific data: what actually happened at a specific race, not what a track is generally like).

These are **written for this project**, not copied from any source. Unlike the race recaps, circuit characteristics are real-world, stable facts rather than results from this project's synthetic 2026 season, so they didn't need live fastf1 verification the way race recaps did - the one exception is the round hosting this season's "Bahrain Grand Prix" (round 16), whose real location per `fastf1_client.get_season_schedule` is Kuala Lumpur, not Bahrain - that file is written for the actual Kuala Lumpur venue and flags the naming discrepancy explicitly, rather than describing the real Bahrain International Circuit under a name this calendar doesn't actually use for it.

Kept as a **separate Chroma collection** from `race_recaps` (both live in the same `RAG_PERSIST_DIR`, just under different collection names) rather than merged into one corpus - race recaps are narrow and race/season-specific, while track profiles are broad and circuit-general; mixing them would likely worsen the retrieval-precision issue already noted in the race-recaps README (vague queries pulling in the wrong chunk), since a track profile mentioning a driver in passing could out-compete the actually-relevant race recap for a query, and vice versa.

| File | Circuit | Location |
|---|---|---|
| albert_park.md | Albert Park Circuit | Melbourne, Australia |
| shanghai.md | Shanghai International Circuit | Shanghai, China |
| suzuka.md | Suzuka International Racing Course | Suzuka, Japan |
| miami.md | Miami International Autodrome | Miami Gardens, United States |
| montreal.md | Circuit Gilles Villeneuve | Montréal, Canada |
| monaco.md | Circuit de Monaco | Monte Carlo, Monaco |
| barcelona.md | Circuit de Barcelona-Catalunya | Barcelona, Spain |
| red_bull_ring.md | Red Bull Ring | Spielberg, Austria |
| silverstone.md | Silverstone Circuit | Silverstone, United Kingdom |
| spa_francorchamps.md | Circuit de Spa-Francorchamps | Spa-Francorchamps, Belgium |
| hungaroring.md | Hungaroring | Budapest, Hungary |
| zandvoort.md | Circuit Zandvoort | Zandvoort, Netherlands |
| monza.md | Autodromo Nazionale Monza | Monza, Italy |
| madrid.md | Madring (Madrid circuit) | Madrid, Spain |
| baku.md | Baku City Circuit | Baku, Azerbaijan |
| kuala_lumpur.md | Kuala Lumpur Circuit | Kuala Lumpur, Malaysia (hosts this calendar's "Bahrain GP") |
| marina_bay.md | Marina Bay Street Circuit | Singapore |
| austin.md | Circuit of the Americas | Austin, United States |
| mexico_city.md | Autódromo Hermanos Rodríguez | Mexico City, Mexico |
| interlagos.md | Autódromo José Carlos Pace | São Paulo, Brazil |
| las_vegas.md | Las Vegas Strip Circuit | Las Vegas, United States |
| lusail.md | Lusail International Circuit | Lusail, Qatar |
| yas_marina.md | Yas Marina Circuit | Abu Dhabi, United Arab Emirates |

Each file has a YAML frontmatter block (`circuit`, `location`, `country`, `track_type`) that the ingestion script reads into vector store metadata - no `season`/`round`/`date` fields, since these aren't season-specific documents.
