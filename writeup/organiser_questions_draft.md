# Draft email to the challenge organisers

Subject: Three rules-clarification questions (parameter count / pretrained teacher / gallery)

Hi,

We have three questions about the 5,000,000-parameter and "provided data only" rules. Each
one materially changes our approach, so we would rather ask than guess.

**1. Pretrained model as a distillation teacher.**
May we train a large *pretrained* model (e.g. a ViT) as a teacher — used *only* during
training, never shipped — and distil its representations into our ≤5M submitted model, with
the teacher itself fine-tuned/adapted on *only the 11,758 provided training images* (no
external images, no external labels)? The submitted model would contain no teacher weights
and would be under 5M parameters. The rules state pretrained weights count toward the budget,
which we read as pretrained *initialisation* being permitted; we are unsure whether a teacher
that never ships is treated the same way.

**2. Parameter count for re-parameterisable architectures.**
For architectures such as RepViT / FastViT that are over-parameterised at training time and
algebraically fuse to a smaller inference graph, is the 5,000,000 limit measured on the
*deployed / fused* model or on the training-time model?

**3. Retrieval-gallery embeddings.**
Our method retrieves against embeddings of the provided training images at inference. These
are cached feature vectors (data derived from the provided images), not learned weights. Do
they count toward the 5,000,000-parameter budget?

Thanks very much,
[name]
