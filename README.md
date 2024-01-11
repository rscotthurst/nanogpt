# NanoGPT Implementation - Generate Shakespeare
Scott Hurst

## Introduction
I wrote this code (not copy/pasted) while following [this video](https://www.youtube.com/watch?v=kCc8FmEb1nY). By downloading the [raw complete works of Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) and training a transformer on it, I'm able to generate new text based on the input corpus.

This is not going to look great, the model is just trying to recreate patterns it sees to generate infinite Shakespeare. In fact the best I get while following the video is something like this:

RICHARD:
Be self more and to Harry, no:
Ah, heard that a Romeo scurfolk slavish word:
And that himself thought to record a clowledge.
Will it pack of the truth vex your oath,
And enforces these advice these my scourts
Within blashock down steps of your loyars.

KING HENRY VI:
Well, what me? That lack-trived belove them?
O God, well'd had! may not with this consul I,
I cannot be some night. Ay, farewell.

## Files
- **`bigram.py`:** This file contains the full model, optimized for GPU. The first few lines contain configurable hyperparameters but the rest should run without changes.
- **`environment.yaml`:** Here I've frozen the working conda environment if needed in the future. To install run `conda env create -n pytorch_p310 -f environment.yaml`.
- **`gpt_dev.ipynb`:** This notebook is a result of following the first part of [this video](https://www.youtube.com/watch?v=kCc8FmEb1nY). This explains with examples how various concepts of transformer architecture work. This notebook will run without a GPU.s 
- **`input.txt`:** The [raw complete works of Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) used as input to the model.
- **`sentence_completion.ipynb`:** A prior attempt to build a sentence completer, it's fun to play around with but not super useful.

## Commits
Because building `bigram.py` was an iterative process, I tried to match up my commits with sections of the video to "freeze" my progress along the way.

- **[Simplest bigram model, mostly replicates what is in Jupyter notebook but in script form](https://github.com/rscotthurst/nanogpt/commit/9fe29510bb480c21afc5715f867ffa4d7522a356)**: [video chapter](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2280s)
- **[Implement single-headed self attention](https://github.com/rscotthurst/nanogpt/commit/38fd33f6e65c8ef5d2475b4b7ef49ab0f370796d)**: [video chapter](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4751s)
- **[Implement multi-headed self attention](https://github.com/rscotthurst/nanogpt/commit/d54194da090c1f78c86b8c5ff9ab1d2b145a069a)**: [video chapter](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4751s)
- **[Add a feedforward layer](https://github.com/rscotthurst/nanogpt/commit/009feeb3b69ac91a66dca9d35f4ddb2a4c33cdd1)**: [video chapter](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5065s)
- **[Block optimization 1: add residual pathway](https://github.com/rscotthurst/nanogpt/commit/19f8a9125fbd74644e724a22f8a3b2284f480cd6)**: [video chapter](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5208s)
- **[Block optimization 2: add layer norm](https://github.com/rscotthurst/nanogpt/commit/56576a1b49b647978741dadbfae6fa5f0ff3b0d2)**: [video chapter](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5571s)
- **[Add dropout to reduce overfitting](https://github.com/rscotthurst/nanogpt/commit/73ec432e6c4d7120242164bf27ad17259825fae9)** [video chapter](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5869s)
- **[Scale up hyperparams massively to run](https://github.com/rscotthurst/nanogpt/commit/85259b430440e80049304e5685e1a8dc6112fb5d)**: [video chapter](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5869s)


## Running Full Model
The notebook `gpt_dev.ipynb` should work fine on any computer. But `bigram.py` should be run with a GPU. I've been using a `ml.g4dn.2xlarge` instance to run it.

Running is pretty straightforward:

```bash
bash  # switch to bash shell
cd SageMaker/nanogpt  # chgdir into appropriate directory
tmux new-session -s nanogpt  # optional but helpful; this can run for a while
conda activate pytorch_p310  # activate torch conda env
python bigram.py  # run bigram.py
```

If `tmux` is being used type `CTRL + B`, then `D` to exit and return to shell. The process will run in the background. When you want to return to check on it, use `tmux attach -t nanogpt`.
