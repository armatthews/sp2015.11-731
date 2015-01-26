First I compound split the German side of the data, and ran the Snowball stemmer
on both sides of the data. I also lower cased both sides of the data.

I implemented the HMM alignment model by Vogel et al., with a few minor changes.
First, instead of using the max approximation, I trained with full forward-backwards.
Second, I allowed target words to align to NULL.

Next, I ran a Bayesian version of the fast align model described in Dyer et al. (2013).
Alignments in each direction were initialized with the output of the HMM model above.
I ran a Gibbs sampler for 1000 iterations with a burn-in of 500 iterations, then
computed the mode of the remaining iterations.

Next, I symmetrized the alignments using the grow-diag heuristic.
Finally, I deleted links between numbers and non-numbers, as well as
between punctuation and non-punctuation tokens. Some extra care had to
be taken to restore the alignments to be compatible with the original,
un-compound-split German.

Short result summary: (P/R/AER)
HMM forward: 70.1% 76.8% 26.9%
HMM backward: xx.x% xx.x% xx.x%
HMM intersection: 92.5% 64.9% 23.3%
BFA forward: 72.8% 80.1% 23.9%
BFA backward: 71.9% 75.3% 26.5%
BFA grow-diag: 78.3% 81.9% 20.0%
BFA grow-diag + postedit: xx.x% xx.x% 18.98%

For the HMM, intersection proved to be the best symmetrization, and postediting did not help.
