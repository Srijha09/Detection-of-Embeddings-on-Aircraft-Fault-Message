# word2vec-non-NLP-
Word vectors are simply vectors of numbers that represent the meaning of a word. For now, that’s not very clear but, we’ll come back to it in a bit. It is useful, first of all to consider why word vectors are considered such a leap forward from traditional representations of words.Word vectors represent words as multidimensional continuous floating point numbers where semantically similar words are mapped to proximate points in geometric space. In simpler terms, a word vector is a row of real valued numbers (as opposed to dummy numbers) where each point captures a dimension of the word’s meaning and where semantically similar words have similar vectors.

we can add and subtract vectors — the canonical example here is showing that by using word vectors we can determine that:

king - man + woman = queen
In other words, we can subtract one meaning from the word vector for king (i.e. maleness), add another meaning (femaleness), and show that this new word vector (king — man + woman) maps most closely to the word vector for queen.

In the Word2Vec model, the objective is to compute conditional probabilities of the form P(w|c), where w is a word and c is the context, or P(c|w). In the analysis of text, the context (c) is often the set of words surrounding w.

 In the analysis of text, the context (c) is often the set of words surrounding w. In our proposed framework, the MERIT model, the words can be FDEs and the context can be MMSGs, or vice-versa.
 
