# LLM Priority Router vs Dictionary Router Diff Analysis

This note analyzes why the priority-rule LLM router differs from the final
dictionary evidence router.

## Summary

- Total routed items: 500
- Route changed vs dictionary: 112
- Changed by dataset:
  - ScanRefer: 40
  - NR3D: 72

## Main Transition Patterns

| Dataset | Dictionary route | LLM route | Count |
|---|---|---|---:|
| NR3D | spatial-only text | E0 | 24 |
| NR3D | E0 | 3D position text | 23 |
| ScanRefer | E0 | spatial-only text | 19 |
| NR3D | E0 | spatial-only text | 14 |
| ScanRefer | E0 | 3D position text | 10 |
| ScanRefer | spatial-only text | E0 | 8 |
| NR3D | spatial-only text | 3D position text | 5 |
| NR3D | E0 | BEV | 4 |

## Performance Impact On Changed Cases

| Dataset | Changed N | LLM Acc@0.25 | E0 Acc@0.25 | Net vs E0 |
|---|---:|---:|---:|---:|
| ScanRefer | 40 | 0.450 | 0.600 | -6 |
| NR3D | 72 | 0.653 | 0.653 | 0 |

The major harmful transition is ScanRefer `E0 -> spatial-only text`.

| Dataset | Transition | N | LLM Acc@0.25 | E0 Acc@0.25 | Net |
|---|---|---:|---:|---:|---:|
| ScanRefer | E0 -> spatial-only text | 19 | 0.316 | 0.632 | -6 |
| ScanRefer | E0 -> BEV | 2 | 0.000 | 0.500 | -1 |
| ScanRefer | 3D position text -> spatial-only text | 1 | 1.000 | 0.000 | +1 |

## Root Causes

1. The dictionary router uses query-type metadata as a hard gate.
   If `query_type == proximity_derived`, it always routes to spatial-only text
   before checking visual attributes. The LLM instead reinterprets the sentence
   semantically and sometimes sends proximity-labeled queries back to E0 when
   visual words are present.

2. The LLM over-expands proximity cues.
   It often treats words such as `by`, `near`, `close to`, `beside`, `corner`,
   `to the left of`, and `by the desk` as rule-1 proximity. In ScanRefer, many
   such examples were intentionally kept in E0 by the dictionary router because
   they are mixed visual/spatial descriptions or broad direction queries.

3. The LLM over-expands geometric cues.
   It sends several `explicit_direction` or mixed visual queries to 3D position
   text because they contain `under`, `above`, `between`, or similar geometry
   words. The dictionary router only uses 3D position text when the parser label
   is `geometric` and the query has no visual attribute cue.

4. Dictionary visual-attribute detection is literal and conservative.
   The dictionary treats terms such as `white`, `black`, `rectangular`, `metal`,
   `plastic`, `on top`, and object-state words as E0 evidence. The LLM sometimes
   ignores those visual requirements if a spatial cue appears earlier or looks
   more semantically important.

5. NR3D changes are less harmful in this recomposition setup.
   Many NR3D route changes either fall back to E0 because the requested source
   output is unavailable, or happen on cases where E0 and the selected source
   agree at Acc@0.25. This is why NR3D has many route changes but zero net loss
   on the changed subset.

## Representative ScanRefer Regressions

| Case | Dictionary | LLM | Query |
|---:|---|---|---|
| 52 | E0 | spatial-only text | the tiny mini fridge . the mini fridge on the corner by the whiteboard . |
| 94 | E0 | spatial-only text | the picture is affixed to the wall . it is located to the left of the door . |
| 101 | E0 | spatial-only text | the cool office chair . the chair is by the desk . |
| 134 | E0 | spatial-only text | the trash can is on the floor to the left of the cabinet . there is a small white bin beside it , to the right . |
| 182 | E0 | BEV | this is a rectangular flat panel monitor . it is the last monitor on the right in the front row of monitors . |

## Interpretation

The priority prompt made the LLM more structured than the free-choice prompt,
but it still does not exactly reproduce the dictionary router because the LLM
uses semantic judgment while the dictionary uses hard parser-label gates plus a
small literal visual-attribute dictionary.

For the paper ablation, this is a useful result: replacing the deterministic
router with an LLM is not automatically better. The LLM can over-route mixed
visual/spatial ScanRefer queries away from E0 unless the prompt or output schema
forces explicit evidence flags such as `is_proximity_main`, `has_visual_evidence`,
and `is_pure_ordinal`.
