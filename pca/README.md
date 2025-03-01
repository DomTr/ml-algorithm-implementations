PCA - Principal Component Analysis
It is a dimensionality reduction method that is often used to reduce the dimensionality of large datasets by transforming a large set of variables into a smaller one that still contains most of information in the large set.

For example, given points $y_1, \dots, y_m$ in $\mathbb{R}^p$
, we need to approximate them by other points whose span has $dim \leq d$

If a dataset of 100'000 peoples' genes was given where each person would be a row vector with 100 types of genes we're interested in, then maybe it would be possible to find some d eigenpeople s.t. other peoples' genes would be possible to be written as a linear combination of eigenpeople genes. Then only the coefficients of linear combinations would be saved, there are d such coefficients in total (how much to take of eigenperson1, eigenperson2, etc.)

This will reduce the dimensionality of the dataset. To preserve as much information as possible, we will take the first left-singular vector matrix $U_d$.