# Transformer Gallery

Implementation of transformer papers from scratch without any specific prebuilt functions.

The purpose of this repository is to understand and benchmark different transformer variants on long sequence benchmarks.

We also aim to quantify different abilities in the long sequence regime such as memory, reasoning, and training speed.

## Transformer Variants

### Transformer

First proposed in "Attention is all you need" paper, transformer is now the backbone of natural language processing.

### Transformer-XL

Transformer-XL stands for Transformer Extra Long, it caches past hidden states at eacb layer to be used as keys and values for the next step. Therefore, information can be propagated forward significantly increasing the context length.

### Longformer

Longformer uses sliding-window attention such that the memory requirements scale linearly instead of quadratically. Attention is performed in overlapping chunks instead of the entire sequence, much like a convolutional neural network. Information outside of the chunks are propagated through deeper layers like a CNN.

### Memorizing Transformer

Memorizing Transformer modifies a layer such that it stores key and value pairs into a kNN search index. In the memorizing layer, self-attention and cross-attention is performed where the self-attention queries is used to search the most similar keys via kNN and cross-attention is done with the queries and retrieved key/value pairs.

### Block Recurrent Transformer

Block Recurrent Transformer proposes a recurrent layer that maintains a recurrent state like a RNN. In the recurrence layer, 2 self attention and 2 cross attention is performed with 4 different sets of queries from the inputs and recurrent state. BPTT is performed inside the sequence by dividing the it into windows.


## Citations

