from pathlib import Path
import re

root=Path(__file__).resolve().parent.parent
p=root/'script.md'
s=p.read_text(encoding='utf-8')
(root/'.slides-build/script.original.md').write_text(s,encoding='utf-8')
# Shift references before inserting the five new sections.
s=re.sub(r'(?i)(slide )(\d+)', lambda m:m[1]+str(int(m[2])+5 if int(m[2])>=9 else int(m[2])),s)
s=re.sub(r'(?m)^\| (11|12|14) \|', lambda m:'| '+str(int(m[1])+5)+' |',s)
s=s.replace('Prepared for the **15 slides in Fashion-MNIST Comparison.pdf**, using **report.pdf** for the supporting explanations and results.', 'Prepared for the **20 slides in [Fashion-MNIST Comparison - Expanded.pdf](<Fashion-MNIST Comparison - Expanded.pdf>)**. Slides 9-13 add one deeper explanation per model after the original overview. Architecture and new-slide results were checked against this checkout\'s pipeline and saved run summaries.')
s=s.replace('**15-17 minutes**','**20-23 minutes**')
begin=s.index('## Slide 8 -')
end=s.index('## Slide 14 -',begin)
overview='''## Slide 8 - The five models

**Point to:** The five model cards from left to right.

**Say:**

> These five models use the same image values but different structures: flat pixels for Linear and MLP, local filters for CNN, row memory for LSTM, and patch attention for Transformer.
>
> Their parameter counts differ, so this is not a comparison at matched capacity. The next five slides explain exactly how each implementation produces its ten class scores.

'''
new=(root/'.slides-build/new_script_sections.md').read_text(encoding='utf-8')
s=s[:begin]+overview+new+'\n'+s[end:]
s=s.replace('Keep slides 7-10 and the confusion-matrix explanation as the technical core.', 'Keep slide 7, the brief overview on slide 8, slides 14-15, and the confusion-matrix explanation on slide 18 as the technical core. For a shorter talk, summarize slides 9-13 in one sentence each using the table below.')
s=s.replace('On slide 8, omit individual layer widths if necessary, but keep the distinction between flat pixels, local filters, row memory, and patch attention.', 'On slide 8, keep the introduction brief. On slides 9-13, use these short explanations when time is limited:')
needle='On slide 18, keep the matrix orientation'
s=s.replace(needle,'''\n\n| Slide | Short model explanation |
|---|---|
| 9 | “Linear learns ten weighted sums of the 784 pixels, with no hidden layer. It gives our 84.26 percent baseline.” |
| 10 | “MLP adds hidden layers of 256 and 128 units, each with ReLU and training dropout, to learn nonlinear pixel combinations.” |
| 11 | “CNN uses shared local filters and pooling, then classifies 64 seven-by-seven maps with a dense head. Most of its parameters are in that head.” |
| 12 | “LSTM reads 28 rows, resets memory for each image, and classifies only its final 128-dimensional hidden state.” |
| 13 | “Transformer embeds 49 patches, applies two four-head attention blocks, then normalizes and averages all tokens before classification.” |

On slide 18, keep the matrix orientation''')
start=s.index('## Source and consistency notes - preparation only')
s=s[:start]+'''## Source and consistency notes - preparation only

- Presentation order: [expanded 20-slide PDF](<Fashion-MNIST Comparison - Expanded.pdf>). The original 15 pages are preserved, and the five model slides appear as slides 9-13. The separate [five-slide PDF](<Fashion-MNIST Model Deep Dives.pdf>) contains only the additions.
- Architecture authority: [models.py](../../a1/models.py), including all five standalone classifiers. [data.py](../../a1/data.py) defines normalization and the flat, image, row, and patch layouts. [train.py](../../a1/train.py) defines Adam, cross-entropy, validation-loss selection, and checkpoint reload.
- Recorded classification results and parameter counts: [comparison.json](../../runs/m2/comparison/comparison.json), backed by each model's `summary.json` and `test_summary.json` under `runs/m1` (Linear/MLP) and `runs/m2` (CNN/LSTM/Transformer). These are saved results, not new training or inference in this slide-editing task.
- The standalone Transformer implementation and run directory are present in this checkout. Earlier script notes saying they were absent are obsolete. Its encoder uses sequence-first tensors internally, pre-norm sublayers, final layer normalization, token mean pooling, and a dropout readout.
- Timing values retained on original slides and in this script come from the supplied comparison PDF. Its Linear/MLP CUDA timing values refer to separate runs, while the original accuracy checkpoints used MPS. The separate `runs/cuda_baselines` directory is absent here, so those timing values have not been independently reverified from local timing artifacts. They must not be substituted with the original MPS training times or silently mixed with the original-checkpoint forward timings in `comparison.json`.
- “Leads on every headline metric” on slide 15 is spoken as **“strongest classification results”** because CNN does not lead on speed or parameter count. Slide 7's train-only normalization prevents held-out data from setting the normalization constants; it does not prove the absence of every form of leakage.
- The five new slides use shapes for one image, omitting the batch axis. Dropout is active only during training. Model outputs are logits, and cross-entropy receives them directly.
- The original PDF is unchanged. Only the new slide artifacts and this script were authored; pipeline code and `AI_USAGE.md` were not modified.
'''
p.write_text(s,encoding='utf-8')
headings=re.findall(r'^## Slide (\d+) -',s,re.M)
assert headings==[str(i) for i in range(1,21)],headings
print('Updated script: 20 sequential slide sections')
