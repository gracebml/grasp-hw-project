---
title: GRAT demo
emoji: 🦾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Language-driven grasp detection, step by step
---

# GRAT: Grounding as Transport

Upload an image and an instruction naming an object part, such as *"Lift spoon by its handle."* The page shows how GRAT turns
CLIP's guess of the target into a grasp: the relevance map, the source of the flow, the eight flow steps and the detected
grasp. A toggle compares GRAT with the baseline of the same pipeline.

Code: [github.com/gracebml/grasp-hw-project](https://github.com/gracebml/grasp-hw-project) (folder `demo/`). The Space runs
on CPU with the seed-123 checkpoints of the paper.
