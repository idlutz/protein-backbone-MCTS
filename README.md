# protein-backbone-MCTS
A generative protein design method for assembling alpha helices and loops into optimal protein backbones using provided sets of geometric constraints and score functions, as described in the paper ["Top-down design of protein architectures with reinforcement learning"](TODO:insert_link).

Use this [container](https://files.ipd.uw.edu/pub/protein-backbone-MCTS/protein-backbone-MCTS.sif) (download size 4.6GB) to run the sampling scripts:
```
protein-backbone-MCTS.sif <sampling_script>.py
```
Sampled backbones passing filtering criteria are output in the `outputs/` folder. The python scripts can be modified to customize the geometric constraints and score functions, sampling procedure and parameters, and final filtering criteria in order to generate desired backbones.

The scripts for sampling tetrahedral, octahedral, and icosahedral cages can be run as above; the shape filling, pore closure, and pseudosymmetric cage scripts require additional arguments. For `fill_shapes.py`, provide a `.obj` file specificying the desired shape (e.g. `protein-backbone-MCTS.sif fill_shapes.py example_shapes/test_sphere.obj`). For `pore_closure.py`, a PDB is provided to initialize the sampling (`example_pore_starts/`). For `pseudosym_cages.py`, 

Contact Isaac Lutz (ilutz@uw.edu, isaacdlutz@gmail.com) with any questions or comments.
