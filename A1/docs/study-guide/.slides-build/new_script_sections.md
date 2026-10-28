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

