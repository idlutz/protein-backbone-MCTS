# protein-backbone-MCTS
A generative method for assembling alpha helices and loops into optimal protein backbones provided sets of geometric constraints and score functions, as described in the paper ["Top-down design of protein architectures with reinforcement learning"](insert_link).

Use this [container](https://files.ipd.uw.edu/pub/protein-backbone-MCTS/protein-backbone-MCTS.sif) (download size 4.6GB) to run the sampling scripts:
```
protein-backbone-MCTS.sif <sampling_script>.py
```
Sampled backbones passing filtering criteria are output in the `outputs/` folder. The python scripts can be modified to customize the geometric constraints and score functions, sampling procedure and parameters, and filtering criteria in order to generate desired backbones.

Contact Isaac Lutz (ilutz@uw.edu, isaacdlutz@gmail.com) with any questions or comments.
