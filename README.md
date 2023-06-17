# The Attention Mechanism

** Write intro intuitively explaining attention - alse state that this is from an NLP lense **

## Queries, Keys, and Values

At the heart of attention lies queries, keys, and values.  The terminology is derived from a "key-value" database, which - as the name may suggest - pairs a key with its associated value, e.g. student name (key) and major (value), member ID (key) and date joined (value), business (key) and valuation (value).  Note that a key or a value can be any datatype.

To retreive a value, we initiate a query, which then surveys the keys to see if any of the keys fullfill the query; if a key satisfies the query, the database then returns the key's associated value.  

Sticking with the example of student name as the key and their major as the value, if we queried our database on "John Doe", it would return John's major, phychology.  Importantly, we can only query on keys, not values, so we can't ask our database in its current form to find people by major; if we wanted to do so, we'd have to switch the keys and values so that the student's major is the key and their name is the value.

We will use the following notation to refer to queries, keys, and values:

Database $\textbf{D} = \{ (k_{1}, v_{1}), (k_{2}, v_{2}), ... , (k_{i}, v_{i}), ..., (k_{n-1}, v_{n-1}), (k_{n}, v_{n})  \}$ where each tuple $(k_{i}, v_{i})$ refers to a key, $k_{i}$, and its associated value, $v_{i}$.
A query will be denoted by $q$.

There are much more to queries, keys, and values, but, for the purposes of understanding attention, the information above suffices.  If you are intrested in databases beyond the bounds of attention, or simply want more information, we have provided some resources !!!WHERE!!!.

## The formula for attention

The 



Sources:
https://bi-insider.com/posts/key-value-nosql-databases/#:~:text=A%20key%2Dvalue%20database%20is,be%20queried%20or%20searched%20upon.
https://www.mongodb.com/databases/key-value-database
https://www.educative.io/blog/what-is-database-query-sql-nosql
https://www.youtube.com/watch?v=OyFJWRnt_AY&ab_channel=PascalPoupart

