# NanoGPT Implementation
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