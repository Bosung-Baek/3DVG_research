# Failure and Recovery Cases

These cases are selected from completed result files. Full RGB-canvas figures require the original rendered canvas assets; this standalone bundle stores query/output evidence and available BEV assets.

## e0_fail_spatial_success / scanrefer case 1

- Query type: `proximity_derived`
- Route: `spatial-only text` (`pure_proximity_spatial`)
- Query: a counter / desk where a tv often sits . it is beside the black chair .
- E0: IoU=0.0000, Acc@0.25=False
- Routed: IoU=0.6584, Acc@0.25=True
- Transition: `recovery`

## e0_fail_spatial_success / scanrefer case 3

- Query type: `proximity_derived`
- Route: `spatial-only text` (`pure_proximity_spatial`)
- Query: there is a long rectangular bench . it is next to another wide bench that 's closer to the pillar .
- E0: IoU=0.0000, Acc@0.25=False
- Routed: IoU=0.9451, Acc@0.25=True
- Transition: `recovery`

## e0_fail_spatial_success / scanrefer case 129

- Query type: `proximity_derived`
- Route: `spatial-only text` (`pure_proximity_spatial`)
- Query: a white pillow . next to the black pillow .
- E0: IoU=0.0161, Acc@0.25=False
- Routed: IoU=0.9268, Acc@0.25=True
- Transition: `recovery`

## e0_fail_spatial_success / scanrefer case 146

- Query type: `proximity_derived`
- Route: `spatial-only text` (`pure_proximity_spatial`)
- Query: this chair is kept closer to the table . on right a white board can be seen on the wall .
- E0: IoU=0.0157, Acc@0.25=False
- Routed: IoU=0.9954, Acc@0.25=True
- Transition: `recovery`

## spatial_regression / scanrefer case 54

- Query type: `proximity_derived`
- Route: `spatial-only text` (`pure_proximity_spatial`)
- Query: this picture is hanged on the wall besides another picture . two monitors are present on the table under these pictures .
- E0: IoU=0.3763, Acc@0.25=True
- Routed: IoU=0.0000, Acc@0.25=False
- Transition: `regression`

## spatial_regression / nr3d case 1

- Query type: `proximity_derived`
- Route: `spatial-only text` (`pure_proximity_spatial`)
- Query: select the sink farther from the washer
- E0: IoU=1.0000, Acc@0.25=True
- Routed: IoU=0.0031, Acc@0.25=False
- Transition: `regression`

## spatial_regression / nr3d case 2

- Query type: `proximity_derived`
- Route: `spatial-only text` (`pure_proximity_spatial`)
- Query: this cabinet is dark and farther away from the door
- E0: IoU=1.0000, Acc@0.25=True
- Routed: IoU=0.0235, Acc@0.25=False
- Transition: `regression`

## spatial_regression / nr3d case 7

- Query type: `proximity_derived`
- Route: `spatial-only text` (`pure_proximity_spatial`)
- Query: the chair with nothing colorful around it
- E0: IoU=1.0000, Acc@0.25=True
- Routed: IoU=0.0000, Acc@0.25=False
- Transition: `regression`

## bev_success / nr3d case 40

- Query type: `ordinal`
- Route: `BEV labeled layout` (`pure_ordinal_bev`)
- Query: the toilet in the right stall
- E0: IoU=1.0000, Acc@0.25=True
- Routed: IoU=1.0000, Acc@0.25=True
- Transition: `both_correct`

## bev_success / nr3d case 85

- Query type: `ordinal`
- Route: `BEV labeled layout` (`pure_ordinal_bev`)
- Query: the smallest bathroom stall
- E0: IoU=1.0000, Acc@0.25=True
- Routed: IoU=1.0000, Acc@0.25=True
- Transition: `both_correct`

## bev_success / nr3d case 99

- Query type: `ordinal`
- Route: `BEV labeled layout` (`pure_ordinal_bev`)
- Query: facing the tables the second table from the left
- E0: IoU=0.0000, Acc@0.25=False
- Routed: IoU=1.0000, Acc@0.25=True
- Transition: `recovery`

## bev_success / nr3d case 195

- Query type: `ordinal`
- Route: `BEV labeled layout` (`pure_ordinal_bev`)
- Query: the cup in the middle
- E0: IoU=0.0082, Acc@0.25=False
- Routed: IoU=1.0000, Acc@0.25=True
- Transition: `recovery`

## router_failure / scanrefer case 9

- Query type: `ordinal`
- Route: `BEV labeled layout` (`pure_ordinal_bev`)
- Query: a chair sits pulled under a table . it 's the third chair from the left .
- E0: IoU=0.0000, Acc@0.25=False
- Routed: IoU=0.0000, Acc@0.25=False
- Transition: `both_wrong`

## router_failure / scanrefer case 16

- Query type: `proximity_derived`
- Route: `spatial-only text` (`pure_proximity_spatial`)
- Query: the chair is next to the northeast - most table . the chair is black with four legs .
- E0: IoU=0.0000, Acc@0.25=False
- Routed: IoU=0.0000, Acc@0.25=False
- Transition: `both_wrong`

## router_failure / scanrefer case 47

- Query type: `proximity_derived`
- Route: `spatial-only text` (`pure_proximity_spatial`)
- Query: there is a white drying rack . placed next to another one .
- E0: IoU=0.0000, Acc@0.25=False
- Routed: IoU=0.0000, Acc@0.25=False
- Transition: `both_wrong`

## router_failure / scanrefer case 48

- Query type: `proximity_derived`
- Route: `spatial-only text` (`pure_proximity_spatial`)
- Query: the grey cushioned chair has wooden arms on each side . this grey cushioned chair has another beside it on the left .
- E0: IoU=0.1461, Acc@0.25=False
- Routed: IoU=0.1461, Acc@0.25=False
- Transition: `both_wrong`

## visual_fallback_kept_correct / scanrefer case 0

- Query type: `explicit_direction`
- Route: `E0 RGB canvas` (`visual_attribute_default_e0`)
- Query: the chair is west of the left - most table . the chair is dark brown and has four legs .
- E0: IoU=0.9332, Acc@0.25=True
- Routed: IoU=0.9332, Acc@0.25=True
- Transition: `both_correct`

## visual_fallback_kept_correct / scanrefer case 2

- Query type: `ordinal`
- Route: `E0 RGB canvas` (`visual_attribute_default_e0`)
- Query: the trash can is right of the right - most door on the northern wall . the trash can is a trapezoidal prism and gray .
- E0: IoU=1.0000, Acc@0.25=True
- Routed: IoU=1.0000, Acc@0.25=True
- Transition: `both_correct`

## visual_fallback_kept_correct / scanrefer case 10

- Query type: `opposite_derived`
- Route: `E0 RGB canvas` (`visual_attribute_default_e0`)
- Query: the monitor is opposite the desk with two monitors , and is next to the white shelf / cabinet . there is a black , non - rolling chair at the same desk .
- E0: IoU=1.0000, Acc@0.25=True
- Routed: IoU=1.0000, Acc@0.25=True
- Transition: `both_correct`

## visual_fallback_kept_correct / scanrefer case 12

- Query type: `explicit_direction`
- Route: `E0 RGB canvas` (`visual_attribute_default_e0`)
- Query: this is a black , rolling office chair . this chair sits at the desk , at the end of the bed .
- E0: IoU=1.0000, Acc@0.25=True
- Routed: IoU=1.0000, Acc@0.25=True
- Transition: `both_correct`
