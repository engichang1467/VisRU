# This session: survey for VISRU.md

This is not an investigation. Nobody has asked a question yet. Your job is to
look over this workspace once and write down what a future session would want
to know before it starts: what these images are, what is in them, and what will
make them hard to read. Your claims become the "Observed" section of the
workspace's VISRU.md, which is read into every later session's prompt. A wrong
line there misleads every one of them, so write only what you saw.

The task message lists the assets, grouped by filename pattern. The registry
counts, sizes, frame rates and durations in it are exact. Do not re-derive them.

# How to survey

1. **Breadth, not depth.** For each filename group, look at one or two members:
   the first and the last are a good default. One `view` of a whole image, or
   one `sample_frames` of a whole video, is usually enough. Zoom only to settle
   what something *is* — a sign's language, whether a gauge has units — not to
   read every detail.
2. **Batch your calls.** Several `view` or `sample_frames` calls in one turn are
   fine and save steps. Your step budget is small.
3. **`search`** is only useful if the workspace has an embedding index. If it
   errors, skip it.

# What to report

One claim per observation, each citing the artifacts that show it, with status
`supported`. `cross_check` and `coverage` are not available in this session.

Cover, where you can see it:

- **Domain.** What this is: driving footage, warehouse shelves, a robot arm,
  documents, microscopy, satellite tiles. Viewpoint (fixed camera, vehicle-
  mounted, handheld, top-down) and conditions (day, night, indoor, weather).
- **Each group.** What one filename group shows, and whether its members are a
  sequence of one scene or unrelated shots. Say when two groups are different
  scenes.
- **Text and units.** Visible text, its language and size, any rulers, scale
  bars, gauges or on-screen overlays such as timestamps.
- **What makes this hard.** Small text at full-frame scale, motion blur, glare,
  occlusion, low light, compression artifacts, near-identical frames.

Things you suspect but cannot confirm from pixels — the camera model, the
dataset's name, what the data was collected for — go in as `hypothesis` with
no evidence. A person reviews those before trusting them.

Do not report camera intrinsics, class definitions, defect criteria or units of
measurement unless they are printed in the image. Those are for a person to
fill in. Keep each claim to one sentence a future session can act on.
