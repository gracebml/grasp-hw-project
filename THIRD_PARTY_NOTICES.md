# Third-party notices

## LGD

`src/lgdbench/fixed_models.py` adapts the two `forward` methods of `inference/models/lgrconvnet3.py` and
`inference/models/lggcnn.py` from [LGD](https://github.com/Fsoft-AIC/LGD), under this license:

```
MIT License

Copyright (c) 2024 An Vuong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Submodules

`ref_sources/LGD` and `ref_sources/CROG` are git submodules: this repository records their URL and commit and holds none of their
files. Each keeps its own MIT license in its own repository.

## Data

The example image in `assets/grat_arch.png` and `visualization/` (`data/example_image.png`, `data/example_mask.png` and the tiles
built from them) comes from the [Grasp-Anything](https://airvlab.github.io/grasp-anything/) dataset (MIT).
