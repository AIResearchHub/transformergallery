# The Attention Mechanism

** Write intro intuitively explaining attention - alse state that this is from an NLP lense **

## Queries, Keys, and Values

At the heart of attention lies queries, keys, and values.  The terminology is derived from a "key-value" database, which - as the name may suggest - pairs a key with its associated value, e.g. student name (key) and major (value), member ID (key) and date joined (value), business (key) and valuation (value).  Note that a key or a value can be any datatype.

To retreive a value, we initiate a query, which then surveys the keys to see if any of the keys fullfill the query; if a key satisfies the query, the database then returns the key's associated value.  

Sticking with the example of student name as the key and their major as the value, if we queried our database on "John Doe", it would return John's major, phychology.  Importantly, we can only query on keys, not values, so we can't ask our database in its current form to find people by major; if we wanted to do so, we'd have to switch the keys and values so that the student's major is the key and their name is the value.

We will use the following notation to refer to queries, keys, and values:

Database $\textbf{D} = \lbrace(k_{1}, v_{1}), (k_{2}, v_{2}), ... , (k_{i}, v_{i}), ..., (k_{n-1}, v_{n-1}), (k_{n}, v_{n})\rbrace$ where each tuple $(k_{i}, v_{i})$ refers to a key, $k_{i}$, and its associated value, $v_{i}$.
A query will be denoted by $q$.

There are much more to queries, keys, and values, but, for the purposes of understanding attention, the information above suffices.  If you are intrested in databases beyond the bounds of attention, or simply want more information, we have provided some resources !!!WHERE!!!.

## The General Formula for Attention

The general formula for attention is as follows:

$$\textrm{Attention}(q, \textbf{D}) = \Sigma_{i}\textrm{ similarity}(q, k_{i})v_{i}$$ where $\textbf{D} := \lbrace(k_{1}, v_{1}), (k_{2}, v_{2}),...,(k_{n}, v_{n})\rbrace$, and $q$ is a query.

In mathmatical terms, attention is a linear combination of the similarity of queries and keys multiplied by the key's associated value for all 

First, let's compare our new attention formula to searching a database using queries, keys, and values.  

When searching a database, we often look for absolute matches, meaning the query and key are identical (similarity = 1).  If we have one student named "John Doe" in our registrar, when we query our database for "John Doe", we expect the similarity of the query and keys to be 0 everywhere except when the key is "John Doe".  Therefore, we'd have a image:

** 1D vector with query John Doe on dimention, and all the keys with a 1 hot for JD **

We'd also call this query/key vector a "one-hot matrix".  Intuitively, we can think of this version of query/key relationship as just a true/false one.

Now consider a similarity score between queries and keys that is not binary but in a range of \[0,1\] rather than an element of {0,1} (note that in pratice, we can have our similarity scores live in any range).

Based on our formula, the more similar the query is to a key - i.e., the closer $\textrm{similarity}(q, k_{i})$ is to 1 - the more the key's associated value will consribute to attention.  Inversely, if $\textrm{similarity}(q, k_{i})$ is small, the attention won't be as impacted by $v_{i}$. 

As an example to build intuition, assume we have a dataset of engeineering major's projecting starting salaries (value) given their GPA (key) (for the sake of the exercise, let's assume that there is some kind of correlation between GPA and starting salary).  

A fourth-year engineering student with a 3.5 (query) might look through the database and focus only on the GPAs in the range of 3.4-3.6 (similar keys), and would expect to receive a starting salary in that range (weighting values more heavily based on query/key similarity) cetrus paribus.  
They might pay some attention to those with a 3.0 or 4.0 GPA (dissimilar keys) and the starting salaries of those students, but our 3.5 GPA student wouldn't really expect a 3.0 or 4.0 GPA student's salary (weighting values lighly based on query/key correlation), though it still remains a possibility.

This example begs the question: "What exactly is $\textrm{similarity}(q, k_{i})$"?  How will we knoe that it works for the example.  The answer is somewhat unsatisfying: we have to decide what $\textrm{similarity}(q, k_{i})$ is.

Our similarity function can take many forms depending on the task at hand; the burden falls on the practitioner to select the most apporpriate for the task.  Here are some common weighting mechanisms:

** INSERT CHART OR SOMETHING**

### Sources:
https://bi-insider.com/posts/key-value-nosql-databases/#:~:text=A%20key%2Dvalue%20database%20is,be%20queried%20or%20searched%20upon.
https://www.mongodb.com/databases/key-value-database
https://www.educative.io/blog/what-is-database-query-sql-nosql
https://www.youtube.com/watch?v=OyFJWRnt_AY&ab_channel=PascalPoupart

