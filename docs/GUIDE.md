# Guide to this project

Written so that someone with no technical background can follow what this does
and why, and so that I can pick it up again in six months and remember why
every decision was made.

Read Part 1 on its own if you only want to know what it is. Parts 3 onwards get
progressively more technical, and each tool is explained before it is used.

---

## Part 1. What this is, in plain words

Imagine you film a short video walking around your living room on Monday. On
Friday somebody moves the armchair, takes away a cardboard box, and leaves a
rucksack by the door. You film the room again.

This project takes those two videos and tells you:

- the armchair moved, and roughly how far
- the box is gone
- a rucksack appeared
- the table did not move

It also shows you this as a 3D model you can spin round in a web browser, with
the things that changed picked out in colour.

Nobody has to tell it what to look for. It works out the shape of the room from
the videos themselves.

### Why that is harder than it sounds

If you simply compared the two videos frame by frame, almost everything would
look different. You stood in a different spot. You held the phone at a
different angle. You walked a slightly different route. The light changed.

The genuinely hard part is not spotting that something is different. It is
being certain that an empty space means **the object was taken away** rather
than **I just did not point the camera there the second time**.

A person does this without thinking. A computer has to be told the difference,
and most systems get it wrong: they report everything the camera missed as a
change. That is the problem this project is built around.

---

## Part 2. Why it exists

It was built as a technical challenge for a job application to Humanoid, a
London company building humanoid robots. Their brief was to compare two phone
recordings of a room and identify meaningful 3D changes, while distinguishing
genuine change from differences in camera viewpoint, occlusion and incomplete
observation.

### Why a robot would want this

A robot working in a warehouse or a home builds a map of its surroundings. The
world then moves: pallets get shifted, doors close, boxes appear in corridors.
The robot has to notice.

The consequence of getting it wrong is not symmetrical. If the robot wrongly
thinks a chair is still there, it goes around a chair that is not there, which
wastes a few seconds. If it wrongly thinks a corridor is clear because it never
actually looked, it drives into something. So a system like this should prefer
to say "I do not know" over guessing, and that principle runs through the whole
design.

---

## Part 3. Every tool used, and why

Each one gets a plain explanation first, then what it does in this project, then
why it was chosen over the alternatives.

### COLMAP, through pycolmap

**What it is.** Software that works out the 3D shape of a scene from ordinary
2D photographs, and where the camera was standing for each one. The technique
is called structure from motion. It is the same idea your eyes use: two slightly
different views of the same thing let you judge depth.

**What it does here.** It takes all the frames from both videos and produces two
things: the position and angle of the camera for every frame, and a sparse cloud
of 3D points marking corners and edges it recognised in multiple frames.

**Why this one.** It is the standard tool for the job, it is free, and
`pycolmap` installs with pip so there is no separate program to set up. A robotics
team will know it immediately, which matters when the work is being assessed.

### Depth Anything V2

**What it is.** A neural network that looks at a single photograph and estimates
how far away every pixel is. It was trained on an enormous number of images, so
it has learned what makes things look near or far: perspective, relative size,
how objects overlap.

**What it does here.** COLMAP gives a few hundred accurate 3D points per frame,
which is nowhere near enough to tell whether a chair is in a corner. This fills
in a depth estimate for every pixel.

**Why this one.** It runs on a laptop without a dedicated graphics card. The
textbook alternative, dense multi-view stereo, is more accurate but effectively
needs a powerful GPU.

**Its weakness, stated honestly.** It produces depth that is correct in relative
terms but has no idea of actual scale. Solving that is described in Part 4.

### CLIP

**What it is.** A model from OpenAI trained on hundreds of millions of image and
caption pairs. Give it a picture and a list of possible descriptions, and it
scores how well each description matches.

**What it does here.** Once the geometry has found a lump of space that changed,
CLIP looks at a cropped photo of that space and picks the best match from a list
of household objects, so the output says "a chair" rather than "cluster 3".

**Why this one.** The geometry has already found *where* the object is, so a
full object detector is not needed, only a name. CLIP is small, runs on a
laptop, and needs no training.

### DBSCAN, from scikit-learn

**What it is.** An algorithm that finds groups in scattered data. Unlike most
clustering methods you do not tell it how many groups to expect; it finds
however many there are and labels loose scattered points as noise.

**What it does here.** After the comparison, there are thousands of individual
cells of space marked as changed. DBSCAN groups ones that touch into objects, so
"four thousand changed cells" becomes "a chair-sized lump here and a box-sized
lump there".

**Why this one.** Not knowing the number of objects in advance is the whole
point, and its ability to discard scattered noise is exactly what is needed,
since measurement error produces scattered noise.

### OpenCV

**What it is.** The standard library for working with images and video in code.

**What it does here.** Reads the videos, pulls out frames, resizes them, and
measures how blurry each one is so the bad ones can be thrown away. It also
draws the synthetic test room described in Part 6.

### Plotly

**What it is.** A charting library that produces interactive graphics viewable
in a web browser.

**What it does here.** Produces `scene.html`, the 3D model you can rotate.

**Why this one.** The output is a single file that opens in any browser. The
usual 3D alternative, Open3D, opens a desktop window, which is no use when
somebody is reviewing your work from a link.

### Matplotlib

**What it is.** The standard Python library for charts.

**What it does here.** Draws the figures in the README, from the real output of
a real run rather than by hand, so they cannot drift away from the truth.

### NumPy

**What it is.** The library that lets Python do arithmetic on millions of
numbers quickly.

**What it does here.** Everything numerical. The free-space calculation alone
handles tens of millions of points, which would take hours in ordinary Python
and takes seconds in NumPy.

---

## Part 4. What actually happens, step by step

### Step 1. Turn videos into pictures

Sample a couple of frames per second, shrink them, and throw away blurry ones.
Blur is measured by how sharp the edges are. A blurry frame is worse than no
frame, because the next stage relies on recognising the same corner in several
pictures, and blur destroys corners.

### Step 2. Work out the shape of the room, from both videos at once

**This is the most important decision in the project.**

The obvious approach is to build a 3D model from video A, build another from
video B, then line them up. That approach asks the hardest question first:
lining up two 3D models works by finding the position where they best agree, and
it is least reliable exactly when part of the scene has genuinely moved. The
things that changed drag the alignment off, and then everything looks changed.

So instead, every frame from both videos goes into **one** reconstruction. The
walls, the floor and the furniture that stayed put appear in both recordings, so
the software matches them across the two videos by itself and everything comes
out in a single shared coordinate system. There is no lining-up step to get
wrong.

Put simply: **the parts of the room that did not change are what tie the two
visits together.** The stable scene is the evidence; the changes are the thing
being measured.

The code refuses to continue if both videos did not end up in the same model,
because a model containing only video A would report the whole of video B as new
and that would look like a result.

### Step 3. Fill in the depth, and fix the scale

The sparse points from Step 2 are accurate but sparse. Depth Anything fills in
every pixel, but its output has no scale: it knows the sofa is nearer than the
wall, not that it is two metres away.

The fix uses one against the other. Both describe the same frame, so finding the
stretch and shift that makes the estimated depths line up with the accurate
sparse points converts the whole dense estimate into the same units as the
reconstruction. One small calculation per frame.

Frames with too few sparse points to do this reliably are skipped rather than
guessed at. A wrongly scaled depth map does not give a slightly wrong answer, it
confidently places objects in the wrong part of the room.

There is one further wrinkle. Reconstruction from photos alone can never recover
true real-world size, because a doll's house photographed close up looks the
same as a real house photographed from further away. So the whole scene is
rescaled once so that typical distances come out room-sized. This is not true
measurement, but it makes every setting in the configuration file mean roughly
what it says.

### Step 4. Record what each visit knows, including what was empty

Space is divided into a grid of cells, each five centimetres across. For each
visit, every cell is marked as one of three things:

| Label | Meaning |
|---|---|
| **occupied** | something was measured here |
| **free** | a camera ray passed through here on its way to hitting something further away, so this space was looked at and was empty |
| **unknown** | neither, so nothing can be said about it |

**Free is the one that makes this work.** Finding it means tracing the line from
the camera out to each thing it saw, and marking everything along the way as
confirmed empty. It is the expensive part of the whole program and it is the
reason the results can be trusted.

The tracing stops slightly short of the surface. Without that, the front of
every object would erase itself and solid things would dissolve.

### Step 5. Compare

Two rules:

```
something was REMOVED if it was occupied in visit A and confirmed empty in visit B
something was ADDED   if it was occupied in visit B and confirmed empty in visit A
```

Both need positive evidence from the other visit. A cell that held something in
A and was simply never looked at in B produces nothing at all.

That one restriction is the difference between a system that reports what
changed and one that reports where the camera went.

### Step 6. Group, name, and pair up

The changed cells are grouped into objects. Each group is then traced back into
the photographs that saw it, cropped out, and shown to CLIP for a name.

One subtle detail: a removed object is cropped from a **visit A** photo, because
by visit B it is not there any more. Getting that backwards would return a
picture of whatever is sitting in that space now, label it confidently, and
produce output that is wrong while looking entirely reasonable.

Finally, a disappearance in one place and an appearance in another are reported
as a single **move** only if they agree on name, are a similar size, and are not
too far apart. Otherwise they stay as separate events, because that is the
honest reading.

---

## Part 5. The decisions that mattered

**Reconstructing both visits together rather than aligning two models.**
Covered in Step 2. The single most important choice.

**Tracking empty space, not just full space.** The thing that stops occlusion
being reported as change.

**Refusing to answer when the evidence is missing.** The system will miss real
changes in parts of the room only one visit covered. That is intended. For a
robot, a missed change is a wasted detour; a hallucinated clear path is a
collision.

**Refusing to name things it cannot see properly.** An early version
confidently announced that a television had moved. It had not. CLIP was
guessing, at a confidence barely distinguishable from random. There is now a
floor: anything scoring near chance is reported as "unidentified object", and
moves are not paired on a name nobody believes. The output went from six
invented findings to ten honest ones.

**Testing against a known answer.** Rather than filming a room and eyeballing
the output, the project includes a program that draws a fake room where the
changes are scripted in advance. That makes the results checkable. All four
scripted changes are found, in the right places, at sensible sizes.

---

## Part 6. The test room

`tools/make_synthetic_room.py` draws two videos of an imaginary room from a
moving viewpoint. Between the two, one box moves across the room, one is taken
away, one is added, and one stays put.

Because the answer is known in advance, the output can be marked rather than
admired.

The room is covered in colourful noise rather than looking realistic, and that
is deliberate: the reconstruction works by recognising the same distinctive
patch in several frames, and a smooth, evenly coloured surface gives it nothing
to grip. A realistic-looking render would reconstruct badly for reasons that
have nothing to do with the system being tested.

The honest limitation is that CLIP cannot name a cube covered in noise, so every
label on the synthetic run comes back as "unidentified object". The fake room
tests the geometry. Only a real room can test the naming.

---

## Part 7. Glossary

| Term | What it means |
|---|---|
| **Structure from motion** | Working out 3D shape and camera positions from a set of 2D photos |
| **Point cloud** | A 3D shape stored as a lot of individual points in space |
| **Sparse / dense** | Sparse means a few hundred accurate points. Dense means a depth value for every pixel |
| **Voxel** | A cube-shaped cell of 3D space, the 3D equivalent of a pixel. Here, five centimetres across |
| **Occupancy map** | A record of which cells of space contain something, are known to be empty, or were never seen |
| **Ray casting** | Tracing the straight line from the camera out through a pixel, to find what the camera was looking through |
| **Occlusion** | One object hiding another from the camera |
| **Monocular depth** | Estimating distance from a single image, rather than from two views |
| **Affine-invariant depth** | Depth that is correct in relative terms but has unknown scale and offset |
| **Zero-shot classification** | Naming something using a model that was never specifically trained on that list of names |
| **Clustering** | Grouping scattered measurements into objects |
| **SIFT keypoint** | A distinctive small patch of an image that can be recognised again from a different angle |
| **Pose** | Where the camera was and which way it was pointing |

---

## Part 8. Questions worth being ready for

**Why not just compare the two point clouds?**
Because absence in a point cloud is ambiguous. It can mean the object left or
that the camera never looked. On real footage, where nobody walks the identical
route twice, the second case is most of the room. Tracking confirmed empty space
separates them.

**Why one reconstruction instead of aligning two?**
Aligning two models means solving for the transform that makes them agree, and
that is least reliable when part of the scene has moved, which is the only case
that matters. Doing it jointly lets the unchanged scene do the aligning.

**How do you know it works?**
A synthetic room with scripted changes. All four are found, in the right places,
at volumes consistent with the surface area of the objects. The figures in the
README are generated from that run, not drawn.

**What are the weaknesses?**
Monocular depth is the weakest link, and its errors are smooth and plausible,
which is the worst kind. There is no true metric scale. Six false positives show
up on the test scene, from depth error on flat walls. And the labelling is
untested on real objects.

**What would you do next?**
Real metric scale from a known-size marker or the phone's own motion tracking.
Dense stereo where there is a GPU. Log-odds occupancy so confidence builds up
across frames instead of a hard threshold. An open-vocabulary detector so
objects outside the list can be named.

**What did you get wrong along the way?**
Three things, all caught by testing. The depth limit was written in metres while
the reconstruction used arbitrary units, so every measurement was being thrown
away and nothing fused. The first synthetic room had no walls, because any
surface with one corner behind the camera was discarded whole, which was fixed
by cutting surfaces into tiles. And the naming stage reported guesses as fact
until a confidence floor was added.
