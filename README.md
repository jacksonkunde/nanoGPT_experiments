# nanoGPT Experiments 🦉

### Progress 🦉 ⬜ ⬜ ⬜ ⬜

## Step 1: Reproduce Character-Level Shakespeare

**Prepare the Dataset:**
```bash
python data/shakespeare_char/prepare.py
```
> length of dataset in characters: 1,115,394
all the unique characters: 
!$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
vocab size: 65
train has 1,003,854 tokens
val has 111,540 tokens

**Train the Model:**
```bash
python train.py config/train_shakespeare_char.py \
    --device=cpu \
    --compile=False \
    --eval_iters=20 \
    --log_interval=1 \
    --block_size=64 \
    --batch_size=12 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --max_iters=2000 \
    --lr_decay_iters=2000 \
    --dropout=0.0
```
**Sample Outputs From the Model**
```bash
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
> I by doth what letterd fain flowarrman,
Lotheefuly daught shouss blate thou his though'd that opt--
Hammine than you, not neme your down way. 
>
>ELANUS:
I would and murser wormen that more?
>
>DUKE VINCENTIO:
At fet she it to on une, so indamen the the
Wat Perciudo on in a so, be and now,
In woll where the arpleon all arnt me tender.

## Step 2: Hyperparameter Experimentation
