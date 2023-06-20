# Transformer
The transformer is arguably the most influential model of the past decade.  It utilizes the attention mechanism so that the attention scales with the available computation, so that, in theory, a transformer's possible attention is unbounded.

** Continue intro **

## Architecture
If you've ever looked up a tutorial on transformers, you've likely seen this image from the seminal paper "Attention is All You Need" (CITE):

<img width="298" alt="Screen Shot 2023-06-20 at 3 23 50 PM" src="https://github.com/ArjunSohur/transformergallery/assets/105809809/1123363a-f956-450e-abc2-70909c555651">

The problem with this pricture is that, unless you already know the ins and outs of a transformer model, the picture can be very confusing.

In the following article, we will traverse this scary picture and try to make it seem like common sense.

** TALK ABOUT ENCODER DECODER MODEL **

A slight disclaimer: to understand the transformer, you need to understand self-attention and multi-head attention.  Knowledge of them will be assumed and used as a base in the following explanations.  We have an article outlining these attention mechanisms to a sufficient degree of rigor if you would like to learn about it or if you need to burnish your memory.

## Positional Encoding
Before both the encoder and decoder, the input embadding and output embedding must go through a positional encoding step.

Transformers are attention-based models, so they are actually agnostic about each word's position, and, therefore, the potentially important positional relationships between words.

The most simple form of positional encoding is simply assigning a positive integer to each word in the sentence.  This method does designate each word to a unique position, but it starts to get shakey when dealing with sequences of varying sizes.

If the goal is to understand a position's relationship to the whole sentence, then, using this form of positional encoding, we'd be training our algorithm to treat the 3rd word of a 5 word sentnece the same as the 3rd word of a Harry Potter novel.  Clearly, both of these words have varying levels of importance relative to their entire sequence.

Ok, then, what if we normalize that positional encoding score so that for a sequence of $n$ words, we just assign word $i$ to $\frac{i}{n}$?

While this idea takes the a word's position relative to a 



## Encoder

## Decoder

### Sources
https://machinelearningmastery.com/the-transformer-model/
https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/#:~:text=What%20Is%20Positional%20Encoding%3F,item's%20position%20in%20transformer%20models.
https://medium.datadriveninvestor.com/transformer-break-down-positional-encoding-c8d1bbbf79a8




