# load data
multiple logics, each logic with several examples, each example with clean and corrupt, and corresponding answer
new dataset for meta learning
train: domain 1: [logic 1, logic 2] domain 2: [logic 1, logic 3], domain 3: [logic 2, logic 3]
each domain has several examples. 
test: [logic 4, logic 5] [logic 5, logic 6]
How to test if it's right? -> get_item, get batch 10 logic 1, 5 logic 2. (5 +5 -5) 8 * 8
train dataset better be generated at first, for each pair, we have 5 drawal from random samples
for test dataset, for setting: id we derive the scores for all, setting: ood, we derive the scores for only unseen logics. 
output: batch[logic 1, logic 2, logic 3] choose n logics, each has p examples in a batch. each logic will have TOTAL / batch_size groups. Each time, we randomly select 2 groups from n logics. maybe we the total training set size is C^(TOTAL)_n.

# find the circuit
score = attribute()
calculate attribute score 
hook on the attribute, edge attribute score
This should have gradient
Test: get a score from some random examples, and have gradient on that example.
# contrastive learning on the circuit
after having the score. We need to use a simple contrastive learning methods to learn. 
# meta learning 
for the 5 withdrawal of each pair, it's considered a domain in meta-learning. only enable gradient of important part.
Now i don't even enable gradient of gradient, but it's already exploding. 
how can i decrease the memory? -> decrease the precision. use smaller models.
I need at least 4 logics in each forward 
# to improve the memory efficiency
for each contrastive, we only enable layer-wise attribution.
we could put attributions from different logics to different layers.
hook only attached to certain layers.
will this converge? 
 

dataset -> patch -> contrastive learning -> meta-learning -> null-space -> model performance -> edit performance -> distance
