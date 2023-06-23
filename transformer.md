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

The most simple form of positional encoding is simply assigning a positive integer to each word in the sentence.  This method does designate each word to a unique position, but it starts to get shakey when dealing with sequences of varying sizes, which can make training difficult.  If we trained this way, it will make the model flounder on sequences of unseen lengths, since it won't know the significance of a position it has never seen.

Ok, then, what if we normalize that positional encoding score so that for a sequence of $n$ words, we just assign word $i$ to $\frac{i}{n}$?

While this idea takes the word's position relative to the whole sequence into account, it starts getting hairy when you assign .2 to the 1st word of a sequence of 5 and the 50th word of a 250 word long sequence.

The trick is to not avoid assigning each position a number, but instead use a fixed length vector that we can gaurentee will be different in different positions.  The following is an algorithm that returns different vectors given a different position without creating the same vector twice:

To create a positional vector of dimention $d$ for position $pos$ in our sequence (which we will denote as $p_{pos} \in \mathbb{R}^d$), for $i$ s.t. $0 \leq i < d/2$:

$$ (p_{pos})^{(2i)} = \text{sin}(\frac{pos}{n^{2i/d}}) \text{ and } (p_{pos})^{(2i+1)} = \text{cos}(\frac{pos}{n^{2i/d}})$$

Where:
- $(p_{pos})^{(k)} \in \mathbb{R}$ denotes the scalar corresponding $k$th dimention of $p_{pos}$ (or the $k$ th element in the $p_{pos}$ array to use CS verbiage)
- $n$ is some predetermined number; the original transformer authors set it to $n=10000$

Let's think about how is happening for a bit.  We are creating a vector for the arbitrary position $pos$ of our sequence, which we are calling $p_{pos}$.  The values of $p_{pos}$ will be defined two at a time with index variable $i$ (we bound i < d/2$ , where $d$ is the dimention or length of $p_{pos}$, so that we stay in the bounds of $p_{pos}$ since one $i$ will define two values of $p_{pos}$ at a time).

It is important to see how our formula produces entirely different vectors of dimention $d$ for every position.  The trick lies in the behavior of the sinusidal functions and the clever way the formula was set up.  Without launching into a trigonometry lesson, putting $pos$ in the numerator of the angle of the sinusidal function (along with the rest of the formula) guarentees that no two position's formula's are the same.  If you want to play with the variables of the formula, a well as get a better intrinsic understanding as to why the last statement is true, check out this graph:

https://www.desmos.com/calculator/x5qk7e5dut

In the end, we add these positional vectors to our preexisting word embeddings so that our transformer has information about the position of each word built into each vector.

** Questions to answer: why does it alternate functinos for even/odd?  How does the model significance of the positional vectors (how does it know what position a word is in if it just receives a vector with no indication of how much of that vector was influenced by the positional encoding); does it just learn them in traning? **

Now, we should understand the motivation and technique behding this part of the graph:
<img width="514" alt="Screen Shot 2023-06-23 at 5 13 04 PM" src="https://github.com/ArjunSohur/transformergallery/assets/105809809/4c422c71-57cb-4f05-a8b7-55c9d44360ae">

## Encoder

## Decoder

### Sources
https://machinelearningmastery.com/the-transformer-model/
https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/#:~:text=What%20Is%20Positional%20Encoding%3F,item's%20position%20in%20transformer%20models.
https://medium.datadriveninvestor.com/transformer-break-down-positional-encoding-c8d1bbbf79a8
https://kazemnejad.com/blog/transformer_architecture_positional_encoding/




