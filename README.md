# The Attention Mechanism

The Attention Mechanism represents a milestone in the evolution of neural networks, particularly within the realm of Natural Language Processing (NLP). The core idea behind attention is the ability to focus on specific parts of an input when generating an output, enabling the model to pay 'attention' to contextual information that contributes most to understanding the input. Similar to how humans selectively focus on certain parts of an image or conversation, attention mechanisms allow models to associate different levels of relevance or importance to input features when making predictions.

we'll explore the key concepts that underpin the Attention Mechanism, such as queries, keys, and values, which form the basis of the mechanism, and how they relate to the broader concept of attention in the context of NLP. We'll also delve into the mathematical formulation of attention, discussing its general formula, the ways we calculate weights, and different approaches to measuring similarity.

Furthermore, we will introduce self-attention, a variant of the attention mechanism used extensively in state-of-the-art models like Transformers. We will help build an intuitive understanding of self-attention and its mathematical representation.

## Queries, Keys, and Values

At the heart of attention lie queries, keys, and values.  The terminology is derived from a "key-value" database, which - as the name may suggest - pairs a key with its associated value, e.g. student name (key) and major (value), member ID (key) and date joined (value), business (key) and valuation (value).  Note that a key or a value can be any datatype.

To retrieve a value, we initiate a query, which then surveys the keys to see if any of the keys fulfill the query; if a key satisfies the query, the database then returns the key's associated value.  

Sticking with the example of student name as the key and their major as the value, if we queried our database on "John Doe", it would return John's major, psychology.  Importantly, we can only query on keys, not values, so we can't ask our database in its current form to find people by major; if we wanted to do so, we'd have to switch the keys and values so that the student's major is the key and their name is the value.

We will use the following notation to refer to queries, keys, and values:

Database $\textbf{D} = \lbrace(k_{1}, v_{1}), (k_{2}, v_{2}), ... , (k_{i}, v_{i}), ..., (k_{n-1}, v_{n-1}), (k_{n}, v_{n})\rbrace$ where each tuple $(k_{i}, v_{i})$ refers to a key, $k_{i}$, and its associated value, $v_{i}$.
A query will be denoted by $q$.

<img src="https://github.com/ArjunSohur/transformergallery/assets/44383158/1fd0357e-73b4-4b4b-bd36-cf22bf5acb51"  width="40%" height="40%">

There is much more to queries, keys, and values, but, to understand attention, the information above suffices.  If you are interested in databases beyond the bounds of attention or simply want more information, we have provided some resources !!!WHERE!!!.

## The General Formula for Attention

The general formula for attention is as follows:

$$\textrm{Attention}(q, \textbf{D}) = \Sigma_{i}\textrm{ similarity}(q, k_{i})v_{i}$$ where $\textbf{D} := \lbrace(k_{1}, v_{1}), (k_{2}, v_{2}),...,(k_{n}, v_{n})\rbrace$, and $q$ is a query.

In mathematical terms, attention is a linear combination of the similarity of queries and keys multiplied by the key's associated value for all keys.

First, let's compare our new attention formula to searching a database using queries, keys, and values.  

When searching a database, we often look for absolute matches, meaning the query and key are identical (similarity = 1).  If we have one student named "John Doe" in our registrar, when we query our database for "John Doe", we expect the similarity of the query and keys to be 0 everywhere except when the key is "John Doe".  Therefore, we'd have:

<img src="https://github.com/ArjunSohur/transformergallery/assets/44383158/0edb657f-e9b7-4702-9ad2-008b09480085"  width="80%" height="80%">

We'd also call this query/key vector a "one-hot matrix".  Intuitively, we can think of this version of the query/key relationship as just a true/false one.

Now consider a similarity score between queries and keys that is not binary but in a range of \[0,1\] rather than an element of {0,1} (note that in practice, we can have our similarity scores live in any range).

Based on our formula, the more similar the query is to a key - i.e., the closer $\textrm{similarity}(q, k_{i})$ is to 1 - the more the key's associated value will contribute to attention.  Inversely, if $\textrm{similarity}(q, k_{i})$ is small, the attention won't be as impacted by $v_{i}$. 

As an example to build intuition, assume we have a dataset of engineering major's projecting starting salaries (value) given their GPA (key) (for the sake of the exercise, let's assume that there is some kind of correlation between GPA and starting salary).  

A fourth-year engineering student with a 3.5 (query) might look through the database and focus only on the GPAs in the range of 3.4-3.6 (similar keys) and would expect to receive a starting salary in that range (weighting values more heavily based on query/key similarity) cetrus paribus.  
They might pay some attention to those with a 3.0 or 4.0 GPA (dissimilar keys) and the starting salaries of those students, but our 3.5 GPA student wouldn't really expect a 3.0 or 4.0 GPA student's salary (weighting values lightly based on query/key correlation), though it still remains a possibility.

This example begs the question: "What exactly is $\textrm{similarity}(q, k_{i})$"?  How will we know that it works for the example?  The answer is somewhat unsatisfying: we have to decide what $\textrm{similarity}(q, k_{i})$ is.

Our similarity function can take many forms depending on the task at hand; the burden falls on the practitioner to select the most appropriate for the task.  

## Common Weight Calculation Methods:
**Dot Product:** A method where attention weights are determined by calculating the dot product between the query and the key vectors. This aligns the direction of the vectors and emphasizes the similarity.

$$s = q^{T}k$$

**Scaled Dot Product:** An approach where the dot product between query and key vectors is scaled by the inverse square root of the vectors' dimensionality. This helps manage the scale of the values before applying the softmax function. (generally what is used in transformer models)

$$s = \frac{q^{T}k}{\sqrt{d}}$$

**Additive:** A method using a single-hidden-layer feed-forward network to generate alignment scores between the query and key inputs. These scores are then normalized using a softmax function to produce the attention weights.

$$s = v^{T} \tanh(W_{1}q + W_{2}k + b)$$

**Multiplicative:** A technique where the dot product of the query and key vectors is calculated to produce the alignment scores, which are then converted into attention weights using softmax normalization.

$$s = q^{T}Wk$$

**Gating Mechanism:** An advanced approach where an additional gating factor, usually determined by a sigmoid function, is applied to the conventional attention weight calculation method. This allows for dynamic adjustment of attention weights.
$$g = \sigma(W_{g}[q;k])$$
$$s = g \cdot (q^{T}k)$$
Where $W_n$ is a learned weight matrix, and $\sigma$ is the sigmoid function.

**Sparse Attention:** A strategy that selectively determines the calculation of attention weights based on selected positions within the sequence. This results in sparse weighting matrices and improves computational efficiency.

$$s = \sum_{i=1}^{k} q^{T}k_{i}$$

## Self Attention

**Intuition**

Self-attention is perhaps the most complicated and integral part of the transformer model.  Before we explore the math and functions behind it, let's cover some intuition.  

If attention is how much we decide to weight certain features, then self-attention naturally follows as how much we decide to weight our own features.  Let's start with an example:

"When he saw his owner, the dog was _____ , so he wagged his tail"

Try to guess what word best suits the <blank> token.  Chances are, if you know anything about dogs, you know that dogs are happy when they see their owners and that wagging their tail is an expression of their happiness.

Taking certain words into context, like "owner", "dog", and "wagged", we can guess that the <blank> is probably "happy" or "excited".  In this case, let's assume that the correct answer is "happy".

Let's now fit our example into the query, key, value paradigm.  In our sentence:

"When he saw his owner, the dog was happy, so he wagged his tail"

We treat all the separate words as keys, and, continuing our above example, let's have "happy" as our query.  Each query/key pair will get assigned a value, which is essentially how much attention should be paid to that key for a given query.  That was a lot of consecutive words, so let's illustrate it:

<img src="https://github.com/ArjunSohur/transformergallery/assets/44383158/64aee3d6-1db5-458b-b752-82d3a0b465c2"  width="80%" height="80%">

In fact, we should have each word take a turn as the query.  If we do so, we end up with the attention matrix:


What we've just done is map out how important each key (column) is to each query (row).  In other words, each row is a demonstration of how much attention the word in question pays to its neighbors (and itself!)

**Math**

Let's start with the first step and build up to getting a self-attention matrix.


Machines speak numbers, so we need to take our words and put them into vector form.  MORE NEEDED  Thankfully, there are plenty of pre-trained algorithms like GloVe or Word2Vec that can do this "embedding" for us.  

Generalizing a bit, let's say that we have n number of words in our input sentence, and our word-to-vector algorithm returns a vector of d dimensions for each word.  After vectorizing, then, we'd have a matrix $X \in \mathbb{R}^{n \times d}$ that represents the input in numerical form where each row of $X$ is a word of the input.

Next, we need to get our keys, queries, and values matrices.  First, we'll go over the math, then deconstruct why the math makes sense.

For an input $X \in \mathbb{R}^{n \times d}$ and trained weights $W_{Q} \in \mathbb{R}^{d \times d_{Q}},  W_{K} \in \mathbb{R}^{d \times d_{K}}, W_{V} \in \mathbb{R}^{d \times d_{V}}$, where $d_{Q}, d_{K}, d_{V} \in \mathbb{R} \text{ s.t. } d_{Q} = d_{K}$ are the number of columns in their respective weight matrices, $$Q = XW_{Q}, K = XW_{K}, V = XW_{V}$$
Which means that $Q \in \mathbb{R}^{n \times d_{Q}}, K \in \mathbb{R}^{n \times d_{K}}, V \in \mathbb{R}^{n \times d_{V}}$.

Ok, so that might seem like gibberish, but let's decipher what is being said in the math.

The key in understanding what $W_{Q}, W_{K}, W_{V}$ are.  The algorithm learns These weights (magic numbers that make everything work) through training.  

Setting up the weights, we need to make sure that their initialized matrices have $d$ amount of rows (which was the number of columns that $X$ has) for legal matrix multiplication.  The number of columns of $W_{Q}$ and $W_{K}$ must be the same, but otherwise, the number of columns of the weight matrices is a hyperparameter that the user might have to play with.

Notice how $Q$, $K$, and $V$ all have n rows (remember that n was the number of words in the sentence).  We can therefore interpret our trio of matrices as having a query, key, or value vector for each word in the sentence that, through the magic of training, when put through a certain formula, results in the appropriate amount of attention.

But what is this "certain formula"?  Using the already defined $Q$, $K$, and $V$, our self attention is: $$S = f(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_{Q}}})V$$

Let's start with $QK^T$.  Recall how each row of $Q$ and $K$ encodes certain information about one of the words in the sentence.  When we take the transpose of $K$, $K^T$, we are making it so that every column of $K^T$ now contains that information about the word rather than the row.  This rearrangement sets us up nicely for matrix multiplication with $Q$, since, by performing $QK^T$, we will be taking the dot product of $K$'s information about a word with $Q$'s information about another (or the same) word.  We will end up with a matrix that has a row for each word detailing other words' relative importance to it, much like in our dog example above REFERENCE IT.

Next, we'll think about what $\frac{1}{\sqrt{d_{Q}}}$ is and its purpose.  Though $\frac{1}{\sqrt{d_{Q}}}$ is intimidating, it is just a scalar for $QK^T$.  Remember that $d_{Q}$ is the number of columns in $W_{Q}$, and therefore also the number of columns in $Q$.  Also recall that $d_{Q} = d_{K}$, which not only makes the matrix multiplication of $QK^T$ possible but means that we could use $d_{K}$ if we so desired. $\frac{1}{\sqrt{d_{Q}}}$ helps keep the values of $QK^T$ from exploding and also helps with gradient descent in backpropagation.


Softmax is defined as follows: for vector $z \in \mathbb{R}^K$ with elements $z_{i}, \forall i \in \mathbb{Z}^{+} \leq K$, then $$\text{softmax}(z_{i}) = \frac{ \exp{(z_{i})}}{\Sigma_{j=1}^{K} \exp{(z_{j})} } $$
Softmax normalizes the values of $z$ (i.e. $\Sigma_{i}^{K} z_{i}$ = 1), guarantees differentiability for backpropagation, and augments the larger values of $z$ while diminishing the smaller ones, hence the name softmax.

Notice how softmax is only defined for vectors, but we apply it to $\frac{QK^T}{\sqrt{d_{Q}}}$, a matrix, in our self-attention formula.  When we do softmax($\frac{QK^T}{\sqrt{d_{Q}}}$), we are implying that the softmax function is applied row-wise to the matrix, meaning that we replace each row with a soft-maxed version of itself.

The last missing puzzle piece is $V$, our values matrix, which rounds off the formula in the same way that the values do in the regular attention formula. UNSURE OF THE INTUITION HERE.  Note that while softmax($\frac{QK^T}{\sqrt{d_{Q}}}$) $\in \mathbb{R}^{n \times n}$, multiplying by $V$ leaves the product as an element of $\mathbb{R}^{n \times d_{V}}$, which means that $V$'s dimensions are important since they determine the final attention matrix's shape.

After all that work, we have finally justified the formula for self-attention: $$S = f(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_{Q}}})V$$
where $S \in \mathbb{R}^{n \times d_{Q}}$.  You might have noticed that earlier in the article, this same formula was listed under "similarity functions" for our basic attention formula.  Though self-attention has been incredibly influential, it is only one method of calculating attention; you might find another similarity more suitable for your needs.

Often, people will refer to this specific self-attention formula as "scaled dot-product attention"

## Multiheaded attention

Once we've understood self-attention, multi-head attention follows naturally.

Usually, we are not just dealing with words in a sentence, but words in an entire document, or sentences in a document.  We might like to know the attention relationship between a word and a sentence but simultaneously have the attention relationship between the word and the larger paragraph that it belongs to.

To get self-attention on different scales, we employ multiple self-attention formulas ($S = f(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_{Q}}})V$) in parallel with different weights and, therefore, different scopes.

A single self-attention mechanism is called a "head", which gives rise to the name "multi-head attention" when we utilize multiple heads simultaneously.

Notationally, for the $i \text{th}$ head, we denote the $i \text{th}$ head's unique query, keys, and values as $W_{Q}^{(i)}, W_{K}^{(i)}, W_{V}^{(i)}$ respectively.  Some people switch the superscript and the sub-script when writing the $i \text{th}$ attention weights, but, for notational consistency with the previous explanation of self-attention, we will write it as we just presented it.

In a multi-head attention layer, the input $X$ gets fed to the multiple heads which use self-attention to produce a unique attention score, $S^{(i)}$, based on their own weights.  We then concatenate all the $S^{(i)}$ for $i \leq$ the number of heads.  Lastly, we feed the massive attention matrix through a feed-forward neural network, and, voila, we have just passed through a multi-head attention layer.

## Summary
The Attention Mechanism revolutionized the way neural networks process information, making them more context-aware. Drawing parallels from how humans selectively focus on certain aspects of a situation, this mechanism assigns different levels of importance to various input features, thereby enhancing the model's performance in tasks like Machine Translation, Question Answering, and many more.

The fundamentals of attention revolve around queries, keys, and values – terms borrowed from key-value databases. A query surveys the keys in a database, and when a key matches the query, the associated value is returned. In the context of attention, the more a key aligns with a query, the more its corresponding value influences the output. 

One advanced form of attention, known as self-attention, has been successfully used in state-of-the-art Transformer models. Self-attention, in essence, evaluates the importance of each word in a sentence in relation to every other word, including itself. This concept allows models to capture long-range dependencies between words, which greatly improves their understanding of the input text.

### Sources:
https://bi-insider.com/posts/key-value-nosql-databases/#:~:text=A%20key%2Dvalue%20database%20is,be%20queried%20or%20searched%20upon.
https://www.mongodb.com/databases/key-value-database
https://www.educative.io/blog/what-is-database-query-sql-nosql
https://www.youtube.com/watch?v=OyFJWRnt_AY&ab_channel=PascalPoupart
https://www.youtube.com/watch?v=i_pfHD4P_wg&ab_channel=SebastianRaschka
https://theaisummer.com/self-attention/
https://en.wikipedia.org/wiki/Softmax_function
https://www.youtube.com/watch?v=A1eUVxscNq8&t=17s&ab_channel=SebastianRaschka
https://d2l.ai/index.html


