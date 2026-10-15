# Design choices and tradeoffs

The brief asks for changes "in 3D", and for actual changes to be told apart from
differences in camera viewpoint, occlusion and incomplete observation. Those two
requirements drove every decision here.

## 1. Reconstruct both recordings as one problem

The obvious approach is to reconstruct each video separately and then register
the two point clouds. I did not do that, because it asks the hardest question
first. Registration works by finding the rigid transform that best aligns two
clouds, and it is least reliable exactly when part of the scene has genuinely
moved, which is the only case that matters here. The things that changed pull
the alignment around, and a bad alignment makes everything look changed.

Instead every frame from both sessions goes into a single COLMAP problem, with
exhaustive matching so the two sessions are compared against each other rather
than only against their own neighbours in time.

The unchanged parts of the room are visible in both recordings, so COLMAP
matches across the session boundary by itself and both sets of cameras come out
in one coordinate frame. There is no alignment step to get wrong. **The parts of
the scene that did not change are what register the two scans**, which is the
right way round: the stable scene is the evidence, and the changes are the
unknown.

The cost is that exhaustive matching is quadratic in the number of frames. For a
room filmed twice at a couple of frames per second that is fine. For a long
survey it would need a vocabulary tree, which is why the matcher is a config
option rather than hard coded.

This also fails loudly rather than quietly. `sfm.py` checks that both sessions
are present in the final model and stops if they are not, because a
reconstruction containing only session A would report the whole of session B as
new, and that looks like a result.

## 2. Monocular depth, anchored to the sparse points

COLMAP's sparse cloud is accurate but far too thin to say whether a chair is
still in a corner. Dense multi-view stereo would be the textbook answer, but it
effectively needs CUDA and I wanted this to run on a laptop.

So each frame gets monocular depth from Depth Anything V2, which is dense but
affine-invariant: correct up to an unknown scale and offset, in its own units.

The sparse points fix that. They are in the same frame, from the same camera, so
fitting a scale and shift that maps predicted depth onto the sparse depths puts
the dense map into COLMAP's coordinates. Depth Anything predicts inverse depth,
so the fit is done in inverse space where the relationship is genuinely linear
and a plain least squares is the right tool. One fit per frame, then a second
pass after dropping the worst tenth of residuals so a few bad triangulations do
not drag the frame.

Frames with fewer than 25 sparse points are skipped rather than guessed at. A
badly scaled depth map does not produce a slightly wrong answer, it produces a
confident wrong answer somewhere else in the room.

## 3. Three states, not a point cloud

This is the part that answers the occlusion requirement, and it is why the
output can be trusted.

Comparing two point clouds cannot distinguish "the chair is gone" from "I never
pointed the camera at that corner the second time". Both look like absence. Any
system that diffs clouds will report the second as a change, and on real footage
where nobody walks exactly the same route twice, that is most of the room.

So each session is stored as an occupancy map with three states:

| State | Meaning |
|---|---|
| `OCCUPIED` | depth measurements landed in this voxel |
| `FREE` | a camera ray passed through on its way to a surface, so the space was looked at and was empty |
| `UNKNOWN` | neither, so no claim can be made |

![the three states resolving into a claim](figures/occupancy_states.png)

`FREE` is what makes the difference, and it is the expensive part. Building it
means marching every sampled ray from the camera centre to just short of where
it hit something, and marking what it passed through as empty. That is a lot of
points, so rays are subsampled on a pixel stride and processed in chunks.

Rays stop a small margin short of the surface. Without that, the front face of
every object carves itself away and solid things dissolve.

## 4. Change is an intersection, not a subtraction

    removed = occupied in A  AND  free in B
    added   = occupied in B  AND  free A

Both need positive evidence from the other session. Something occupied in A and
merely unobserved in B produces nothing at all. That single choice is the
difference between reporting what changed and reporting where the camera went.

![detected changes by volume](figures/change_summary.png)

On the synthetic scene the four real changes come out as the four largest
clusters. The tail below the median line is depth error on flat walls, not
objects, and it is why `min_cluster_voxels` exists.

It is deliberately conservative. The system will miss real changes in parts of
the room that only one pass covered. I would rather under-report than hand a
robot a map claiming a doorway is clear because nobody looked at it.

## 5. Moves are inferred, never assumed

Geometry alone sees an object vanish in one place and another appear elsewhere.
Calling that a move is an inference, so it has to clear three bars: the same
semantic label, comparable volume, and within a sensible distance. Pairs are
matched greedily by a cost that prefers close, similar objects.

Anything that does not pair stays as a plain removal or addition. When a chair
disappears and a box appears across the room, the honest output is one removal
and one addition.

## 6. Semantics from crops, in the right session

Labels come from projecting each change cluster back into the frames that saw
it, cropping the bounding box from the best view, and scoring with CLIP against
a vocabulary of indoor objects.

The detail that matters: a removed object is cropped from a **session A** frame,
because by session B it is not there. Cropping from the wrong session returns a
picture of whatever occupies that space now, labels it confidently, and the
output becomes nonsense that still looks plausible.

CLIP over crops rather than an open-vocabulary detector is a deliberate
simplification. The geometry has already decided where the object is, so
detection is not needed, only naming. It also keeps the whole thing to two model
downloads and runs on CPU.

## What I would do next

- **Metric scale.** Everything is in COLMAP's arbitrary units, so volumes are
  relative. A known-size marker in frame, or ARKit and ARCore poses from the
  phone, would make the numbers real.
- **Dense stereo where there is a GPU.** Monocular depth is the weakest link;
  its errors are smooth and plausible, which is the worst kind.
- **Carry uncertainty through.** Right now a voxel is free or it is not. Log-odds
  occupancy, as in OctoMap, would let confidence accumulate across frames
  instead of a hard threshold on hit counts.
- **Detector instead of a fixed vocabulary.** Grounded SAM would name things the
  config never listed, rather than snapping them to the nearest word on the list.
