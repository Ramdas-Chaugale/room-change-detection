# Example output: the synthetic room

Produced by:

```bash
python tools/make_synthetic_room.py --out videos/
python run.py --a videos/before.mp4 --b videos/after.mp4 \
              --out output/ --config config_synthetic.yaml
```

- `scene.html` open in a browser and drag to rotate
- `report.md` the written summary
- `changes.json` every cluster with its geometry and labelling evidence

## The input

Three matching moments from each recording, before on the left, after on the
right. The full videos are not committed because they are 10 MB each, but
`tools/make_synthetic_room.py` regenerates them exactly, the seeds are fixed.

![before and after](inputs/before_after.jpg)

Individual frames are in `inputs/`. The textures are deliberately noisy: SIFT
needs something to match on, and a cleanly shaded render would reconstruct
badly for reasons that have nothing to do with the pipeline.

## Ground truth, and what was found

The scene is scripted, so the answer is known.

| Truth | Found | Where it was found |
|---|---|---|
| chair moved from the left to the right | yes, as a removal and an addition | 0.28 m3 removed back left, 0.09 m3 added back right |
| box removed from the right | yes | 0.26 m3 removed, front right |
| bag added on the left | yes | 0.19 m3 added, front left |
| table unchanged | yes, correctly silent | not reported |

All four real changes are found, in the right places, at sensible volumes. A
hollow cube of about 0.8 m3 has roughly 5 m2 of surface, and at a 0.05 m voxel
that is about 0.25 m3 of occupied voxels, which is what came back.

Six smaller clusters are also reported, the largest 0.37 m3. Those are false
positives from monocular depth error on flat textured walls, where the depth
model has little to anchor on. They are the honest cost of using monocular depth
rather than dense stereo.

![where the changes are](../docs/figures/topdown.png)

Seen from above, the detections land in different parts of the room rather than
piling into one corner, which is what a result that is tracking real geometry
looks like.

## Why every label says "unidentified object"

Because the renderer draws cubes covered in coloured noise, and CLIP correctly
has no idea what they are.

The label confidence came back at **0.035 against a chance level of 1/29 =
0.034**. The model was guessing. An earlier version of this code reported those
guesses as fact and produced a report confidently announcing that a television
had moved, which was nonsense that looked like a result.

So there is now a confidence floor. Anything scoring near chance is reported as
"unidentified object", and move matching refuses to pair clusters on a label
nobody believes. That is why this run reports ten separate changes rather than
inventing moves between unrelated objects.

**On real footage this is the part that works differently.** The geometry is
what the synthetic scene validates; the semantics cannot be validated on objects
that are not real objects. Record an actual room and the labels have something
to work with.
