# NLP1 Project 2 results

The unidirectional LSTM with fine-tuned GloVe 2024 WikiGigaword 100d averaged
**0.8446 macro-F1 and 84.48% accuracy** on SST-2 validation. The frozen version
averaged 0.8271 macro-F1 and 82.80% accuracy across the same three seeds.
[Reproduction commands](README.md)

## Dataset and evaluation

All final classifiers used 67,349 official SST-2 training examples and were
scored on the same 872 official validation examples. The Project 1 reference
uses token indicators with a linear SVM. Its validation scores were
**0.7790 macro-F1 and 77.98% accuracy**. [SST-2 dataset](https://huggingface.co/datasets/stanfordnlp/sst2)

## GloVe vocabulary coverage

We measured exact lookups in GloVe 2024 WikiGigaword 100d. Token coverage
counts every occurrence. Type coverage counts each distinct token once.

| Split | Tokenizer | Token coverage | Type coverage | Examples with a missing token |
| --- | --- | ---: | ---: | ---: |
| Train | Whitespace | 98.30% | 90.67% | 9,475 / 67,349 |
| Train | Whitespace + internal hyphens | 99.89% | 98.92% | 725 / 67,349 |
| Train | Internal hyphens + slash to `or` | 99.92% | 99.18% | 503 / 67,349 |
| Validation | Whitespace | 98.64% | 95.11% | 193 / 872 |
| Validation | Whitespace + internal hyphens | 99.92% | 99.67% | 14 / 872 |
| Validation | Internal hyphens + slash to `or` | 99.95% | 99.79% | 9 / 872 |

Splitting `well-acted` exposes two vocabulary entries with vectors. The
classifier experiments use hyphen splitting and preserve slash expressions.
The literal slash rule gave five validation examples full coverage, and
it also turned `and/or` into `and or or`. Remaining misses included
`timewaster` and `thekids`. The split negation token `n't` has a vector.

## Task-specific word similarities

We used eight hand-picked anchors from SST-2 and measured cosine similarity
between pretrained vectors. For nearest neighbors, we searched the 13,540
alphabetic training tokens with exact GloVe entries after hyphen and slash
normalization.

| Anchor | Review-related comparison | Contrast |
| --- | --- | --- |
| `good` | `great` (0.802) | `bad` (0.771) |
| `boring` | `dull` (0.701) | `engaging` (0.481) |
| `funny` | `hilarious` (0.822) | `unfunny` (0.559) |
| `predictable` | `formulaic` (0.666) | `unpredictable` (0.750) |
| `gripping` | `riveting` (0.652) | `tedious` (0.387) |
| `moving` | `touching` (0.576) | `shifting` (0.707) |
| `plot` | `storyline` (0.521) | `soundtrack` (0.321) |
| `thriller` | `suspense` (0.720) | `comedy` (0.749) |

The neighbors of `funny` included `hilarious`, `amusing`, and `witty`.
Those of `thriller` included `drama`, `horror`, and `melodrama`. The
high similarity of `good` and `bad` shows that words with opposite
sentiment can occupy similar contexts. The neighbors `turning`, `move`,
and `moved` show that the motion sense is prominent in the vector for
`moving`, a word reviewers also use to describe emotional impact.

## Fixed eight-epoch encoder comparison

The four encoders used frozen WikiGigaword 2024 100d embeddings, a linear
sentiment head, cross-entropy, and AdamW with learning rate 0.001 and weight
decay 0.01. They trained for eight epochs with batch size 256, seed 42,
and zero dropout. Padding was excluded from pooling and recurrence.

| Model | Sentence representation | Trainable parameters | Macro-F1 | Accuracy |
| --- | --- | ---: | ---: | ---: |
| Project 1 token indicators + linear SVM | Token indicators | — | 0.7790 | 77.98% |
| Embed-and-pool | Mean of word vectors | 202 | 0.7554 | 75.57% |
| GRU | Final hidden state, one layer with 128 units | 88,578 | 0.8263 | 82.68% |
| Unidirectional LSTM | Final hidden state, one layer with 128 units | 118,018 | **0.8449** | **84.52%** |
| Bidirectional LSTM | Concatenated final states, 64 units per direction | 85,250 | 0.8211 | 82.11% |

All three recurrent models scored above mean pooling and the Project 1
reference. The unidirectional LSTM gained 6.59 macro-F1 points over Project 1.
The two LSTMs fed 128 features to the head and differed in trainable
parameter count. Validation loss bottomed out at epoch 4 for the GRU and
epoch 2 for both LSTMs, then finished higher as training loss fell. The
unidirectional LSTM's peak validation macro-F1 was 0.8532 at epoch 7.

## Early stopping with patience

We chose epochs on a stratified development fold drawn from official training
data, with 60,615 fit examples and 6,734 development examples. Identical
tokenized inputs stayed together. We monitored development macro-F1 with
patience 3 and a 16-epoch cap, then refit on all official training examples
for the selected epoch count and evaluated once on official validation.

| Model | Selected epoch | Fixed 8-epoch F1 | Patience-selected F1 | Patience-selected accuracy |
| --- | ---: | ---: | ---: | ---: |
| Embed-and-pool | 15 | 0.7554 | 0.7318 | 73.39% |
| GRU | 14 | 0.8263 | 0.8304 | 83.14% |
| Unidirectional LSTM | 16 | 0.8449 | 0.8280 | 82.80% |
| Bidirectional LSTM | 13 | 0.8211 | 0.7958 | 79.82% |

All four selection runs reached the cap. The bidirectional LSTM met the
patience condition at epoch 16. The selected epochs increased official
validation loss for all three recurrent models. The development fold had
a median length of 7 tokens compared with 19 on official validation,
which limits how well epoch choices transfer between those splits.

## Hyperparameter screening

We tested seven variants with seed 42 and the fixed eight-epoch setup. Each
changed one setting against its matched control.

| Variant | Matched control F1 | Variant F1 | Change in F1 points |
| --- | ---: | ---: | ---: |
| Max pooling | 0.7554 | 0.6558 | −9.95 |
| Sum pooling | 0.7554 | 0.7695 | +1.41 |
| LSTM with frozen GloVe Twitter 100d | 0.8449 | 0.8211 | −2.38 |
| LSTM with frozen WikiGigaword 2014 100d | 0.8449 | 0.8312 | −1.37 |
| GRU with 0.2 head dropout | 0.8263 | 0.8272 | +0.08 |
| LSTM with 0.2 head dropout | 0.8449 | 0.8333 | −1.15 |
| LSTM with fine-tuned WikiGigaword 2024 100d | 0.8449 | **0.8485** | +0.37 |

The same unidirectional LSTM with frozen vectors gives task-specific evidence
for choosing WikiGigaword 2024 over the two alternatives. Validation token
coverage was 99.92% for WikiGigaword 2024, 99.73% for WikiGigaword 2014,
and 98.20% for Twitter. Sum was the strongest pooling rule in this screen.
Head dropout made little difference for the GRU and lowered LSTM macro-F1
at the tested rate.

## Paired seed comparison

We repeated the screening winner and the frozen WikiGigaword 2024 LSTM
with seeds 7 and 123. The standard deviations are sample estimates across
all three seeds.

| Seed | Frozen LSTM macro-F1 | Fine-tuned LSTM macro-F1 | Gain in F1 points |
| --- | ---: | ---: | ---: |
| 42 | 0.8449 | 0.8485 | +0.37 |
| 7 | 0.8153 | 0.8425 | +2.72 |
| 123 | 0.8210 | 0.8428 | +2.18 |
| Mean ± sample standard deviation | 0.8271 ± 0.0157 | **0.8446 ± 0.0034** | **+1.76** |

Fine-tuning improved macro-F1 in each paired seed and increased validation
cross-entropy in each run. The screening and seed repeats reused the same
validation examples, so the spread describes variation across training
seeds on that split.
