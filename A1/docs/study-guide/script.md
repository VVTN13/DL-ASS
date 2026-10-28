# Assignment 1 presentation script: Fashion-MNIST comparison

Prepared for the **20 slides in [Fashion-MNIST Comparison - Expanded.pdf](<Fashion-MNIST Comparison - Expanded.pdf>)**. Slides 9-13 add one deeper explanation per model after the original overview. Architecture was checked against this checkout's pipeline. Scores and selected epochs retain the original PDF's comparison snapshot. Spoken text is in English to match the slides.

**Result provenance:** The original comparison reports MLP accuracy **88.50%**, macro-F1 **0.8846**, selected epoch **8**. The current `runs/m1/mlp` summaries instead identify a separate CUDA baseline run with **88.16%**, **0.8808**, epoch **9**. This script and the added slides retain the original comparison consistently. Do not describe those original MLP scores as a fresh result from the currently stored MLP checkpoint.

**How to use:** Read the **Say** paragraphs. The pointing cues, technical backup, and Q&A are preparation notes, not extra speech. Allow roughly **20-23 minutes**, including short pauses; the shorter route near the end reduces this. Say “our team” or “our implementation” when describing the project; you do not need to imply that you personally wrote every component.

**The sentence to remember:** “The CNN gives the best classification results in our tested configuration, while the MLP offers a useful accuracy-speed trade-off; our experiment does not establish a universal ranking of architectures.”

---

## Slide 1 - Comparing Five Architectures on Fashion-MNIST

**Point to:** The five model names and the research question. The decorative product image is not a Fashion-MNIST sample.

**Say:**

> Good morning. I am presenting Team G-09's comparison of five classifiers on Fashion-MNIST: Linear, MLP, CNN, LSTM, and Transformer.
>
> Our question is how the architecture, together with the way we arrange the same image pixels, affects classification accuracy, computational cost, and the kinds of mistakes the model makes.
>
> I will explain the data, the models, and then the results and their limitations.

## Slide 2 - Problem and objectives

**Point to:** The pipeline along the bottom, then the four evaluation objectives.

**Say:**

> Each input is a 28-by-28 grayscale image, and the task is to assign exactly one of ten clothing labels.
>
> The pipeline is straightforward: preprocess the image, pass it through a model, and obtain ten scores called logits. The class with the highest score becomes the prediction.
>
> An online catalogue motivates the task, although this benchmark does not reproduce all the complexity of real product photographs.
>
> We compare four aspects: predictive performance, computational cost, learning behaviour, and errors. Accuracy tells us how many images are correct. Macro-F1 averages a precision-and-recall score across the ten classes, helping us examine performance beyond the overall correct count.

## Slide 3 - Dataset and data split

**Point to:** The 60,000-image training branch, the 54,000/6,000 split, and the separate official test box. Then gesture to the equal-height class bars.

**Say:**

> Fashion-MNIST provides 60,000 training images and a separate official test set of 10,000 images.
>
> We divide the original training set into 54,000 training and 6,000 validation images. The split is stratified, meaning that it preserves the class proportions, and seed 42 makes the split repeatable.
>
> Training learns the weights and normalization constants. Validation chooses the saved checkpoint. The official test set evaluates that fixed checkpoint; it does not come from the training split.
>
> Each class has 5,400 training, 600 validation, and 1,000 test images. An exact-image hash audit found no duplicates within or across these partitions. That check does not detect near-duplicates or prove that every possible source of leakage is absent.

## Slide 4 - Exploratory data analysis: image examples

**Point to:** Trouser and Bag, then T-shirt/top, Pullover, Coat, and Shirt in the sample grid.

**Say:**

> These examples show why a small image can still be difficult to classify. Trousers and bags have distinctive outlines, while several upper-body garments look similar. Images within the same class also vary in brightness and shape.
>
> About half the training pixels are exactly zero, but no training image is completely blank. We therefore retain the images, and the balanced class counts give us no reason to rebalance the dataset.
>
> The main observation is that the model needs useful visual detail, not just a rough silhouette.

## Slide 5 - Exploratory data analysis: brightness and coverage

**Point to:** The blue brightness distributions, then the orange nonzero-coverage distributions. Do not reread the repeated bullets.

**Say:**

> These violin plots show the distribution of two simple image properties. Brightness is average pixel intensity. Coverage is the percentage of pixels above zero. A wider part of a violin means more images have values in that region.
>
> Coats are brightest on average, at about 98, while sandals are darkest, at about 35, on the zero-to-255 scale. However, the distributions overlap substantially.
>
> So these properties describe the data, but neither provides a reliable rule for separating all ten classes. This motivates learning spatial patterns from the pixels.

## Slide 6 - Exploratory data analysis: class similarity

**Point to:** The Coat-Shirt off-diagonal cell, shown rounded to 0.99. Avoid dwelling on the diagonal.

**Say:**

> Here we average the training images within each class and compare those average images using cosine similarity. Values closer to one indicate more similar spatial patterns.
>
> The diagonal is one because each class is compared with itself. The strongest similarity between different classes is Coat and Shirt, at 0.9873.
>
> This suggests that upper-body garments may be difficult to separate. But this is not a confusion matrix: it describes average images, not model predictions. We will check whether a related pattern appears in the actual errors later.

## Slide 7 - Preprocessing and input representations

**Point to:** The same Bag image in the four layouts. Explain the Transformer plot as 49 rows of 16 patch pixels, not a new image resolution.

**Say:**

> Every model receives the same 784 pixel values after the same numerical preprocessing.
>
> We first divide by 255, then subtract the training mean, approximately 0.2862, and divide by the training standard deviation, approximately 0.3532. Validation and test reuse these constants. Negative standardized values are expected; we do not clip them.
>
> What changes is the layout. Linear and MLP receive a flat vector. CNN keeps the two-dimensional grid. LSTM reads 28 rows, each containing 28 features. Transformer receives 49 non-overlapping four-by-four patches, each flattened into 16 values.
>
> No pixels are discarded, and we use no resizing or augmentation. Reconstruction and consistency checks verify that these layouts preserve the same information. The models differ in how they use that information.

## Slide 8 - The five models

**Point to:** The five model cards from left to right.

**Say:**

> These five models use the same image values but different structures: flat pixels for Linear and MLP, local filters for CNN, row memory for LSTM, and patch attention for Transformer.
>
> Their parameter counts differ, so this is not a comparison at matched capacity. The next five slides explain exactly how each implementation produces its ten class scores.

## Slide 9 - Linear: one weighted score per class

**Point to:** The four-stage diagram, then the parameter calculation on the left.

**Say:**

> Start with the simplest classifier. We flatten the normalized image into 784 values. For each of the ten classes, the model learns one weight per pixel and one bias.
>
> To calculate a class score, it multiplies each pixel by its weight, adds all the products, and adds the bias. That is the equation z equals W x plus b. Ten times 784 weights plus ten biases gives 7,850 parameters.
>
> This model can learn that a bright pixel at a particular location supports a certain class. But it has no hidden nonlinear feature extractor for combining patterns. Its test accuracy is 84.26 percent, which gives us a useful baseline.
>
> The outputs are raw scores called logits. Cross-entropy receives those logits directly during training. At prediction time, we choose the largest score. We do not need a separate softmax for either operation.

## Slide 10 - MLP: nonlinear combinations of pixels

**Point to:** The 256-unit and 128-unit hidden layers, then the dropout explanation.

**Say:**

> The MLP uses the same flat input but adds two hidden layers, with 256 and 128 units. Each dense layer combines its inputs using learned weights.
>
> After each hidden layer, ReLU keeps positive activations and replaces negative ones with zero. This nonlinearity matters: without nonlinear activations, stacked linear layers would still describe a single affine map.
>
> Dropout follows each ReLU, with probability 0.2. It randomly suppresses activations during training, and we disable it for validation and testing. It does not remove images or permanently delete weights.
>
> The final dense layer gives ten logits. The MLP reaches 88.50 percent accuracy, 4.24 percentage points above Linear. This is consistent with the value of richer nonlinear features, but we also increased capacity to 235,146 parameters. The MLP preserves all pixels, while lacking the CNN's shared local filters.

## Slide 11 - CNN: shared filters learn local patterns

**Point to:** The spatial shapes in the diagram, then the 401,536-parameter dense layer.

**Say:**

> The CNN keeps the one-channel, 28-by-28 image grid. Its first convolution learns 32 filters of size three by three. Padding one keeps the spatial dimensions at 28 by 28. ReLU follows, and two-by-two max pooling reduces them to 14 by 14.
>
> The second block repeats this pattern with 64 filters. After pooling, we have 64 feature maps, each seven by seven. A filter reuses the same weights at different locations, allowing it to detect a local pattern across the image.
>
> We flatten these maps into 3,136 features, then use dense 128, ReLU, dropout 0.2, and dense ten. The first dense layer alone contains 401,536 parameters, about 95 percent of the model's total.
>
> The CNN reaches 91.94 percent test accuracy. This is the strongest measured result, but the model also has the most parameters. We need controlled experiments to separate the effect of convolution from model size.

## Slide 12 - LSTM: a memory updated one row at a time

**Point to:** The distinction between 28 steps and 28 pixel features, then the final hidden state.

**Say:**

> For the LSTM, one image becomes 28 steps, with 28 pixel features at each step. We read rows from top to bottom. These are positions within an image, rather than frames in a video.
>
> A single LSTM layer has 128 hidden units and reuses the same weights at every row. The forget gate controls how much old cell memory to keep. The input gate controls how much new content to write. The output gate controls the hidden state exposed from that memory.
>
> Both hidden and cell states begin at zero for every image. After row 28, only the final hidden state goes through dropout 0.2 and the ten-output classifier. We do not average the row outputs or classify the cell state directly.
>
> This gives 82,186 parameters and 88.48 percent accuracy. That is only two correct images behind the MLP on this test set. One seed does not establish a reliable ranking between them.

## Slide 13 - Transformer: patches exchange information

**Point to:** Patch embedding, the encoder blocks, then normalization and mean pooling.

**Say:**

> We split the image into a seven-by-seven grid of four-by-four patches. This gives 49 tokens with 16 pixel values each, in left-to-right, top-to-bottom order. All 784 pixels remain present.
>
> A linear layer projects each patch into 64 features. We add a learned position vector and apply dropout. Positions let the model distinguish where a patch belongs.
>
> Two encoder blocks then process the tokens. Each block uses four attention heads, with 16 features per head. Queries and keys determine how strongly patches attend to one another, and the attention weights mix their values.
>
> Each block also contains a feedforward network with hidden width 128 and GELU. Layer normalization comes before each sublayer, with a residual connection around it. Dropout is 0.2.
>
> Finally, we layer-normalize the tokens, average all 49, apply dropout, and produce ten logits. There is no classification token or causal mask. This model trains from scratch and reaches 85.55 percent within the ten-epoch budget. We will return to that budget limitation in the results.


## Slide 14 - Experimental setup and fairness

**Point to:** Shared training settings, the checkpoint rule, and the timing definition.

**Say:**

> All five configurations use Adam with learning rate 0.001, batch size 128, and ten epochs. They share the split, normalization, and seed. Cross-entropy trains the model to assign a higher probability to the correct class.
>
> After each epoch, we evaluate validation loss and save the weights if it improves. Training still finishes all ten epochs. We then reload the lowest-loss checkpoint for the final test evaluation.
>
> Forward time is averaged over 100 calls after ten warm-ups, using a batch already on the GPU. It excludes loading and transfers.
>
> One qualification: the original Linear and MLP accuracy checkpoints were trained on Apple MPS. Their reported CUDA training times come from separate retraining runs. The report's forward timings all use the GTX 1660 SUPER.
>
> This is a shared training protocol, but not equal capacity or equal training to convergence.

## Slide 15 - Main results

**Point to:** CNN in both bar charts, then MLP and LSTM in the table. Pause after the CNN result.

**Say:**

> The CNN gives the strongest classification results: **91.94 percent test accuracy and 0.9189 macro-F1**. It also has the lowest test loss and makes 806 mistakes out of 10,000 images.
>
> Compared with the MLP, that is a gain of 3.44 percentage points. Errors fall from 1,150 to 806, which is about a 30 percent relative reduction in mistakes.
>
> The MLP improves over Linear by 4.24 percentage points, consistent with the benefit of a richer nonlinear model on the same flat input.
>
> MLP and LSTM are effectively tied in this run: 88.50 versus 88.48 percent, a difference of only two images. One seed does not establish a stable ranking between them.
>
> Transformer reaches 85.55 percent under this configuration. The CNN leads on predictive quality, but we also need to consider learning behaviour and cost.

## Slide 16 - Learning behaviour and computational cost

**Point to:** The MLP's selected-epoch line, the Transformer loss curve, then the runtime table.

**Say:**

> These curves explain why we select checkpoints using validation loss. The MLP is best at epoch eight: its validation loss then increases slightly while training loss continues to fall. Using the last epoch would discard that better checkpoint.
>
> CNN and Transformer select epoch ten. The Transformer is still improving at the budget boundary, so the result does not tell us its best possible performance with longer training.
>
> The CNN's recorded training accuracy is 95.19 percent, compared with 93.28 percent validation accuracy. That gap is a useful generalization check, although training and validation metrics are collected under different conditions.
>
> In the timing table, MLP takes 0.326 milliseconds per batch, CNN 1.074, LSTM 1.189, and Transformer 2.061. These are batches of 128, not single-image request latencies.
>
> LSTM and Transformer have fewer parameters than MLP but take longer. Operation type, sequence processing, and hardware implementation matter as well as parameter count.

## Slide 17 - Error analysis: per-class performance

**Point to:** The Shirt row, followed by the CNN column and the Trouser row.

**Say:**

> The average scores hide large differences between clothing categories. This heatmap gives one F1 score for each model-class combination.
>
> Shirt is the hardest class for every model. Even the CNN reaches only 75.18 percent Shirt F1, compared with 99.10 for Trouser and 98.40 for Bag.
>
> The CNN has the highest F1 in every class in these recorded runs, so its gain is not confined to one category. But strong overall accuracy does not mean every class is equally reliable.
>
> This is consistent with the similar upper-body shapes we observed during EDA. The next slide shows exactly which labels are confused.

## Slide 18 - Error analysis: confusion matrix and shared errors

**Point to:** True Shirt row 6, predicted Coat column 4: 103. Then the reverse cell, row 4/column 6: 24. The 718 on the Shirt diagonal counts correct Shirt predictions.

**Say:**

> Rows show the true labels and columns show predictions. Each row contains 1,000 test images; the diagonal counts correct predictions.
>
> Label six is Shirt and label four is Coat. We see 103 Shirt images predicted as Coat, but only 24 Coat images predicted as Shirt. The confusion is asymmetric.
>
> This explains why Coat recall is 92.7 percent while precision is only 82.77 percent. The model finds most actual coats, but some images it calls Coat belong to other classes.
>
> Overall, 532 of the CNN's 806 errors occur between the four upper-body categories, about 66 percent. Shirt alone contributes 282 errors.
>
> The models also make different mistakes. CNN correctly handles 566 images that MLP misses, while MLP correctly handles 222 images that CNN misses. All five are wrong on 389 shared images. This motivates investigating ensembles, but we have not measured an ensemble result.

## Slide 19 - Limitations and next steps

**Point to:** The limitations column, then the first three next steps.

**Say:**

> Our conclusion is limited to these configurations and recorded runs.
>
> First, one seed per model does not measure variation across training runs. Second, the model sizes differ, so we cannot separate the effect of architecture from capacity. Third, the ten-epoch budget and shared learning rate do not establish the best settings for every model.
>
> Timings also come from separate sessions, and the dataset contains small grayscale images rather than realistic catalogue photographs.
>
> The next steps are repeated seeds, validation-selected longer schedules, and controlled ablations of model size or individual components. Augmentation and confidence calibration are further experiments.
>
> We can say the CNN wins here and Shirt remains difficult. We cannot say convolution alone caused the gain or that attention is generally worse.

## Slide 20 - Conclusion and discussion

**Point to:** The conclusion, then the two discussion questions. Let the audience read the questions rather than reading both long paragraphs verbatim.

**Say:**

> To conclude, the CNN provides the best classification results in our experiment: 91.94 percent accuracy and 0.9189 macro-F1. The MLP remains a useful lower-cost option, while its accuracy is almost identical to the LSTM's. Similar upper-body garments remain the main source of errors.
>
> Our two discussion questions are how to isolate the contribution of spatial structure from model size, and how to choose between accuracy and computational cost for deployment.
>
> Thank you. I am happy to discuss either question.

---

## Technical backup: understand the implementation without memorizing code

Use this section before presenting or when a follow-up question needs more detail. It is not an extension of the read-aloud script.

### The whole pipeline in plain language

1. Load images and integer labels; reuse fixed training and validation indices.
2. Calculate the mean and standard deviation using training pixels only.
3. Standardize each image and arrange its pixels for the selected model.
4. For each training batch: clear gradients, calculate ten logits per image, calculate cross-entropy, backpropagate, and update weights with Adam.
5. After each epoch, turn off dropout and gradient recording for validation. Save an improved validation-loss checkpoint.
6. Finish ten epochs, reload the selected checkpoint, and evaluate the official test set.
7. Save predictions, scores, confusion counts, histories, configuration, and environment information so the result can be inspected.

**Epoch versus batch:** An epoch visits all 54,000 training images. A batch is one group processed for an update. At batch size 128, there are 422 training batches per epoch: 421 full batches and a final batch of 112. The smaller batch is retained. Epoch loss is weighted by sample count.

**Logits versus probabilities:** Logits are unrestricted scores. Softmax converts them into probabilities summing to one. The largest logit and largest softmax probability select the same class. `CrossEntropyLoss` receives logits directly; there is no separate softmax before this loss.

**Adam and learning rate:** Backpropagation calculates how weights affect the loss. Adam uses gradient statistics to adapt updates. The learning rate controls the overall update scale; it stays at 0.001 here because there is no scheduler.

### CNN: exactly what happens to one image

`B` is the batch size; dimensions below describe the whole batch.

| Operation | Output shape | What it does |
|---|---|---|
| Input | `(B, 1, 28, 28)` | One grayscale channel |
| 3x3 convolution, 32 filters, padding 1; ReLU | `(B, 32, 28, 28)` | Learns 32 local feature maps; spatial size stays the same |
| 2x2 max pooling | `(B, 32, 14, 14)` | Retains the strongest response in each pooling window |
| 3x3 convolution, 64 filters, padding 1; ReLU | `(B, 64, 14, 14)` | Combines local features into 64 maps |
| 2x2 max pooling | `(B, 64, 7, 7)` | Reduces spatial dimensions again |
| Flatten | `(B, 3136)` | Arranges `64 x 7 x 7` learned features into a vector |
| Dense 128; ReLU; dropout 0.2 | `(B, 128)` | Combines the features and regularizes training |
| Dense 10 | `(B, 10)` | Produces one score per class |

**Why convolution helps:** A filter uses the same weights at different locations. A useful local pattern therefore does not need a completely separate detector at every pixel position. This is an architectural explanation, not proof that convolution alone caused the measured gain.

**Why so many parameters?** The first dense layer alone has `3136 x 128 + 128 = 401,536` parameters, about 95% of the total 421,642. A CNN is not automatically smaller than an MLP.

**ReLU:** Sets negative activations to zero and introduces nonlinearity. Standardized input values may be negative; preprocessing does not apply ReLU or clip them.

**Dropout:** During training, randomly suppresses a fraction of activations to reduce dependence on particular features. It is disabled for validation and testing. A rate of 0.2 does not mean discarding 20% of images or permanently deleting 20% of weights.

### LSTM: why an image can be treated as a sequence

The shape `(B, 28, 28)` means **batch, 28 row steps, 28 pixel features per step**. The first 28 and second 28 have different meanings.

At each row, the LSTM receives the row and the previous state. Its forget gate controls retained memory, its input gate controls new memory, and its output gate controls the exposed hidden state. These gates are learned from the classification objective.

There is one forward LSTM layer with hidden size 128. Both hidden and cell states start at zero for each image. After all 28 rows, the final hidden state goes through dropout 0.2 and a `128 -> 10` linear classifier. Total parameters: 82,186. There is no bidirectional layer, attention readout, or persistent state between images in this baseline.

**Useful sentence:** “The LSTM summarizes spatial information sequentially; the row index acts like a sequence step, not real time.”

**Why it might struggle:** The final readout compresses the rows into one state and does not directly apply a two-dimensional local filter. That is a plausible interpretation; no row-direction or readout ablation was performed to establish the cause of its score.

### Transformer: how patches communicate

1. Split 28x28 into a 7x7 grid of non-overlapping 4x4 patches: **49 tokens, 16 pixels each**. `49 x 16 = 784`, so patch extraction loses no pixels.
2. A learned linear layer maps 16 pixels to 64 features per token. Add a learned 64-dimensional position vector for each of the 49 locations; apply dropout 0.2.
3. Use two encoder blocks. Each has four attention heads, a 64-dimensional representation, and a feedforward hidden width of 128 with GELU. Layer normalization precedes each sublayer, with residual additions around the sublayers; dropout is 0.2.
4. Apply final layer normalization, average the 49 tokens, apply dropout, and map the 64 features to ten logits. **No CLS token, pretrained backbone, decoder, or causal mask.**

**Queries, keys, values:** A query describes what information a patch is looking for; keys help determine which other patches are relevant; values contain the information to combine. These are learned projections of the token features, not manually assigned labels.

For each head, `attention = softmax(Q K^T / sqrt(16)) V`. The head dimension is `64 / 4 = 16`. The scaling controls the magnitude of similarity scores. Each head can learn different relationships, but the experiment does not establish a named role for each head.

**Why position matters:** Without position information, attention followed by mean pooling would not distinguish permutations of the same patch tokens. Learned positions tell the model where each patch belongs.

**Why it can be slower despite fewer weights:** The same learned operations are applied across tokens, and attention computes token-to-token interactions. Parameter count describes stored learned values, not the number or efficiency of operations in a forward pass.

### Metrics: the definitions you should be able to explain

| Measure | Plain explanation | Example from this experiment |
|---|---|---|
| Accuracy | Correct images divided by all images | CNN: `9194 / 10000 = 91.94%` |
| Recall for a class | Of images truly in this class, how many did we recover? | Shirt: `718 / 1000 = 71.80%` |
| Precision for a class | Of images predicted as this class, how many really belong to it? | Coat: `927 / 1120 = 82.77%` |
| Class F1 | Harmonic mean of that class's precision and recall | `2PR / (P + R)`; Shirt F1 is 75.18% |
| Macro-F1 | Calculate F1 for each class, then average the ten values | CNN: 0.9189 |
| Cross-entropy | Penalizes assigning low probability to the correct label | CNN test loss: 0.2366; lower is better |

Because each class has 1,000 test images, macro recall equals accuracy here, and weighted F1 equals macro-F1. **Macro-F1 still does not equal accuracy**, because F1 also depends on false positives. It is the average of class F1 scores, not F1 calculated from average precision and average recall.

**Percentage points versus relative reduction:** `91.94 - 88.50 = 3.44` percentage points. `(1150 - 806) / 1150 = 29.9%` fewer errors relative to MLP. Do not call the accuracy gain “29.9 percentage points.”

---

## Prepared answers to the two questions on slide 20

### Q1. How would you distinguish spatial structure from model size?

**Answer to say:**

> I would first compare a smaller CNN and an MLP at approximately matched parameter counts, using the same split and training protocol across several seeds. That reduces the capacity difference, although it does not isolate every architectural factor.
>
> To test the value of natural neighbourhoods more directly, I would train the same CNN again after applying one fixed random pixel permutation to every image in all partitions. The permutation preserves the pixel values and model size but disrupts which pixels are neighbours. A consistent performance drop would support the value of natural spatial structure for that CNN.
>
> These would be new controlled experiments, with settings chosen on validation data; they are not results from this report.

**If the lecturer suggests removing pooling:** Removing pooling also changes spatial dimensions, the receptive field, computation, and potentially the dense-layer parameter count. Keep the classifier size controlled or document these changes. Simply removing pooling does not isolate “convolution versus model size.”

### Q2. Would you deploy CNN or MLP, and which metric decides?

**Answer to say:**

> If misclassification cost is the priority and the latency budget permits it, I would start with the CNN because it makes 344 fewer errors on these 10,000 images. If compute cost is tightly constrained, MLP is a candidate because its measured batch forward time is about 3.3 times shorter.
>
> I would choose using the application's error costs and actual end-to-end latency. Macro-F1 gives a useful class-balanced summary, and Shirt recall is an additional requirement if missing Shirts is particularly costly. CNN still misses 282 of the 1,000 Shirts.
>
> Before deployment, I would evaluate real catalogue images and the intended batch size. These GPU batch timings do not establish single-request latency or real-world accuracy.

---

## Likely lecturer questions and concise answers

### 1. What is the main contribution of this assignment?

> A reproducible comparison of five specified configurations under shared data and training settings, combining accuracy and macro-F1 with learning curves, cost, and detailed errors. We are not claiming a new architecture or a state-of-the-art result.

### 2. Why use a Linear classifier if a CNN is expected to do better?

> It gives a simple reference with very few parameters. Comparing MLP with Linear shows the observed gain from adding nonlinear hidden layers and capacity before moving to spatial or sequential models.

### 3. Does flattening destroy information?

> No. The ordering is fixed and all 784 values remain recoverable. What changes is the model's built-in treatment of neighbourhoods: a dense model does not share a local filter across image positions.

### 4. Why divide by 255 and then standardize?

> Dividing by 255 converts integer intensities to a consistent zero-to-one scale. Standardization then centers and scales them using the training distribution. The same constants are reused for validation and test so those partitions do not determine preprocessing parameters.

### 5. Are all standardized images individually mean zero and variance one?

> No. We use one mean and standard deviation across all training pixels. The training pixels collectively have approximately zero mean and unit standard deviation after transformation. Individual images, validation data, and test data need not have those exact statistics.

### 6. Why select the lowest validation loss instead of highest accuracy?

> That rule is fixed in our protocol. Cross-entropy responds to the probability assigned to the correct class, including changes that do not alter the winning label. It therefore provides a more detailed selection signal than accuracy, although it does not guarantee the highest test accuracy.

### 7. Did you use early stopping?

> No. Every run completed ten epochs. We saved improved checkpoints and selected the best one afterwards. An earlier selected epoch is not the same as stopping training early.

### 8. Why can validation loss be lower than training loss?

> Dropout is active during training but disabled during validation. Training metrics also aggregate batches while the weights are changing; validation uses the weights at the end of the epoch. The two losses are measured under different conditions, so this pattern alone is not evidence of leakage.

### 9. Why is Transformer worse here? Would more epochs fix it?

> This is a small Transformer trained from scratch under a short shared schedule. Its validation loss was still decreasing at epoch ten, so longer training is worth testing. We have not shown that it would catch up, and this result does not establish that attention is generally inferior.

### 10. Why did LSTM not beat MLP?

> Reading rows sequentially was not an advantage in this recorded configuration. The final-state readout and the imposed row order are possible explanations, but we did not isolate them experimentally. The measured accuracy gap is only two images.

### 11. Is the comparison actually fair?

> It controls the split, normalization, optimizer, learning rate, batch size, and number of epochs. It does not control parameter count, compute, or convergence. Also, the original Linear and MLP accuracy runs used MPS, while their CUDA training times come from separate runs. It is a useful configuration comparison with those limits.

### 12. Does the fixed seed guarantee the exact same result on another machine?

> It helps reproduce the split, initialization process, and data order, but it does not guarantee identical results across devices, software versions, or kernels. Saved indices, normalization constants, checkpoints, hashes, and environment records make the experiment traceable.

### 13. Is a hash audit proof that the dataset has no leakage?

> It checks exact duplicated image contents and complements the disjoint-index checks. It cannot detect visually similar products, near-duplicates, or every other form of leakage. Computing normalization from training only addresses a separate leakage risk.

### 14. Why report macro-F1 when the dataset is balanced?

> Equal class counts do not imply equal precision and recall. The Shirt class remains much harder than Trouser. Macro-F1 summarizes class-specific precision-recall performance, and the per-class table shows weaknesses that the overall summary still hides.

### 15. Why can Coat have high recall but lower precision?

> The CNN correctly retrieves 927 of the 1,000 actual coats, giving 92.7 percent recall. But it predicts Coat 1,120 times because other garments are also assigned that label. Only 927 of those predictions are correct, giving 82.77 percent precision.

### 16. Does Coat-Shirt cosine similarity prove why the model confuses them?

> No. It measures similarity of class-average images. The confusion matrix describes predictions on individual images. The observations are consistent, but they do not identify the model's internal reasoning or prove a causal explanation.

### 17. Is the CNN's 1.074 milliseconds an image's response time?

> No. It is the mean forward time for a device-resident batch of 128 images. Dividing by 128 gives an amortized per-image figure within that batch, not the latency of an individual request. Loading and transfers are excluded.

### 18. Why does total test evaluation take longer than the forward-only benchmark suggests?

> The complete test pass also loads data, transfers it, calculates loss, and collects predictions. The timers cover different scopes, so their difference is not a clean measurement of loading overhead alone.

### 19. Can combining the models guarantee a better result?

> No. Complementary errors suggest an opportunity, but an ensemble must decide which prediction to trust without knowing the true label. The paired counts do not establish an ensemble score. We would design and select the combination using validation data.

### 20. Does 91.6% confidence mean a prediction is reliable?

> Not necessarily. The report includes a Sandal predicted as Sneaker with 91.6 percent softmax confidence, while a correct Ankle boot has 58.2 percent. We did not evaluate calibration, so the confidence score should not be treated as a verified probability of correctness.

### 21. Why no augmentation, scheduler, or class weighting?

> The comparison uses a deliberately simple shared protocol. Class weighting was unnecessary for balancing because all classes have equal support. Augmentation and schedules might improve results, but those improvements were not measured here.

### 22. Does the CNN training-validation gap prove serious overfitting?

> It shows a difference between the recorded training and validation metrics. The curves and best-checkpoint rule help assess it, but the gap alone does not quantify its cause or severity. Dropout and the timing of metric collection also differ between training and validation.

### 23. What would you improve first?

> Repeat the existing configurations across several seeds to measure variation. Then compare capacity-controlled models and validation-selected longer schedules, especially for Transformer. That would make the comparison more informative before adding more architectural complexity.

### 24. How are the results reproducible?

> The report describes fixed split indices, training-only normalization, saved configurations, checkpoints, epoch histories, per-image predictions, and hash records tying evaluation to the selected checkpoint. These allow someone to inspect the process and recompute classification metrics from predictions. Reproducing training also requires the matching code and environment.

### 25. What if you are asked about a code detail you did not implement?

> “I did not implement that particular component. The documented configuration is [state the relevant setting], and the experiment shows [state the supported result]. I would check the matching implementation before claiming a detail beyond that.”

Use this only when needed. You can confidently explain the documented method and evidence without claiming authorship of every line.

---

## Numbers to keep beside you during Q&A

The forward column is **milliseconds per batch of 128 on CUDA**, not per image. The training column covers the recorded ten-epoch loop. Linear and MLP training times come from the report's separate CUDA timing runs.

| Model | Parameters | Best epoch | Validation accuracy | Test accuracy | Test macro-F1 | Test loss | Errors | Train seconds | Forward ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear | 7,850 | 9 | 86.48% | 84.26% | 0.8417 | 0.4454 | 1,574 | 90.85 | 0.060 |
| MLP | 235,146 | 8 | 89.87% | 88.50% | 0.8846 | 0.3251 | 1,150 | 96.69 | 0.326 |
| CNN | 421,642 | 10 | 93.28% | 91.94% | 0.9189 | 0.2366 | 806 | 105.03 | 1.074 |
| LSTM | 82,186 | 9 | 89.80% | 88.48% | 0.8835 | 0.3254 | 1,152 | 122.12 | 1.189 |
| Transformer | 71,946 | 10 | 86.43% | 85.55% | 0.8524 | 0.4085 | 1,445 | 139.45 | 2.061 |

**Confusion-matrix labels:** 0 T-shirt/top; 1 Trouser; 2 Pullover; 3 Dress; 4 Coat; 5 Sandal; 6 Shirt; 7 Sneaker; 8 Bag; 9 Ankle boot.

**The most useful error facts:** Shirt recall 71.80%, precision 78.90%, F1 75.18%; Shirt-to-Coat 103 versus Coat-to-Shirt 24; 532/806 CNN errors among the four upper-body classes; all five models wrong on 389 images.

**Pronunciation:** CNN: “C-N-N.” MLP: “M-L-P.” LSTM: “L-S-T-M.” ReLU: “ree-loo.” Macro-F1: “macro F-one.” Logits: “loh-jits.” Argmax: “the index of the highest score.”

## Shorter route if presentation time is cut

Keep slide 7, the brief overview on slide 8, slides 14-15, and the confusion-matrix explanation on slide 18 as the technical core. For a shorter talk, summarize slides 9-13 in one sentence each using the table below. Use these substitutions to aim for approximately **10-12 minutes**, depending on pace. They replace the original Say paragraphs for the specified slides.

| Slide | Short replacement |
|---|---|
| 2 | “We classify each 28-by-28 image into one of ten classes. We compare prediction quality, cost, learning curves, and errors, using the highest of ten model scores as the prediction.” |
| 4 | “These samples show distinctive categories such as Trouser and visually similar upper-body garments. The classes are balanced and there are no blank images, so we keep all samples without rebalancing.” |
| 5 | “Brightness and coverage vary by class, but these distributions overlap heavily. Those two simple features alone cannot reliably separate the labels.” |
| 6 | “Coat and Shirt have highly similar average images, with cosine similarity 0.9873. That suggests a challenge to investigate, but this is a prototype-similarity matrix, not prediction errors.” |
| 16 | “The best checkpoint is not always the last: MLP selects epoch eight. Transformer is still improving at epoch ten. CNN takes 1.074 milliseconds per batch of 128, compared with MLP's 0.326. Fewer parameters do not guarantee a faster model.” |
| 17 | “Shirt is hardest for every model. CNN reaches 75.18 percent Shirt F1 but 99.10 for Trouser. This shows why we inspect individual classes as well as overall accuracy.” |
| 19 | “We have one seed, unequal model sizes, and a short shared training budget. The next steps are repeated seeds, capacity-controlled comparisons, and validation-selected longer training. Our result applies to these configurations, not all CNNs and Transformers.” |

On slide 8, keep the introduction brief. On slides 9-13, use these short explanations when time is limited: 

| Slide | Short model explanation |
|---|---|
| 9 | “Linear learns ten weighted sums of the 784 pixels, with no hidden layer. It gives our 84.26 percent baseline.” |
| 10 | “MLP adds hidden layers of 256 and 128 units, each with ReLU and training dropout, to learn nonlinear pixel combinations.” |
| 11 | “CNN uses shared local filters and pooling, then classifies 64 seven-by-seven maps with a dense head. Most of its parameters are in that head.” |
| 12 | “LSTM reads 28 rows, resets memory for each image, and classifies only its final 128-dimensional hidden state.” |
| 13 | “Transformer embeds 49 patches, applies two four-head attention blocks, then normalizes and averages all tokens before classification.” |

On slide 18, keep the matrix orientation and the 103-versus-24 example; omit the paired-model counts if time is running out.

## Source and consistency notes - preparation only

- Presentation order: [expanded 20-slide PDF](<Fashion-MNIST Comparison - Expanded.pdf>). The original 15 pages are preserved, and the five model slides appear as slides 9-13. The separate [five-slide PDF](<Fashion-MNIST Model Deep Dives.pdf>) contains only the additions.
- Architecture authority: [models.py](../../a1/models.py), including all five standalone classifiers. [data.py](../../a1/data.py) defines normalization and the flat, image, row, and patch layouts. [train.py](../../a1/train.py) defines Adam, cross-entropy, validation-loss selection, and checkpoint reload.
- Original comparison results and parameter counts: [comparison.json](../../runs/m2/comparison/comparison.json), matching the supplied PDF. Current `runs/m2` summaries agree for CNN/LSTM/Transformer. Current `runs/m1` summaries identify the separate CUDA baseline runs: Linear retains the same headline scores, but MLP records 88.16% accuracy, macro-F1 0.8808, and selected epoch 9 instead of the original comparison's 88.50%, 0.8846, and epoch 8. These are saved results. A synthetic two-image forward check confirmed the five models return ten logits and match the listed parameter counts; no training or dataset evaluation was repeated.
- The standalone Transformer implementation and run directory are present in this checkout. Earlier script notes saying they were absent are obsolete. Its encoder uses sequence-first tensors internally, pre-norm sublayers, final layer normalization, token mean pooling, and a dropout readout.
- Timing values retained on original slides and in this script come from the supplied comparison PDF. Its Linear/MLP CUDA timing values refer to separate runs, while the original accuracy checkpoints used MPS. Although the `runs/cuda_baselines` directory is absent, the current `runs/m1/linear` and `runs/m1/mlp` summaries identify that output path and confirm the separate CUDA timings (90.85/96.69 seconds training and about 0.060/0.326 ms forward per batch). Their current checkpoint results must remain separate from the original comparison. They must not be substituted with the original MPS training times or silently mixed with the original-checkpoint forward timings in `comparison.json`.
- “Leads on every headline metric” on slide 15 is spoken as **“strongest classification results”** because CNN does not lead on speed or parameter count. Slide 7's train-only normalization prevents held-out data from setting the normalization constants; it does not prove the absence of every form of leakage.
- The five new slides use shapes for one image, omitting the batch axis. Dropout is active only during training. Model outputs are logits, and cross-entropy receives them directly.
- The original PDF is unchanged. Only the new slide artifacts and this script were authored; pipeline code and `AI_USAGE.md` were not modified.
