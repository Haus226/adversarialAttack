### AdversarialAttack Repository Overview

The repository implements several adversarial attack algorithms inspired by various optimization techniques commonly used in machine learning.

Reference: 
[https://github.com/kozistr/pytorch_optimizer/tree/main](https://github.com/kozistr/pytorch_optimizer/tree/main)


#### Types of Attacks

**Targeted Attack:**
A targeted attack aims to make the model misclassify the input as a specific target label. This can be expressed as:

$$
\arg\min_{x'} J(x', y') \text{  s.t  } \|x-x'\|_p\leq\epsilon
$$

where $y'$ is the specific target label.

**Non-Targeted Attack:**
A non-targeted attack seeks to maximize the loss between the adversarial sample and the ground truth label, forcing the model to make any incorrect prediction. This can be formulated as:

$$
\arg\max_{x'} J(x', y) \text{  s.t  } \|x-x'\|_p\leq\epsilon
$$

**Key Differences:**
- **Non-Targeted Attack**: Aims to maximize the loss function between the adversarial example and the true label, causing any incorrect classification.
- **Targeted Attack**: Aims to minimize the loss function between the adversarial example and a specific target label, causing the model to classify the input as the target label.

Despite deep neural network optimizers being primarily designed to minimize the loss, the Momentum and NAG-based methods have shown a higher success rate in adversarial attacks. However, these methods tend to produce perturbations that are more noticeable to the human eye. Conversely, optimizers such as Adagrad, Adadelta, RMSprop, Adam, and NAdam perturb images in smaller steps, requiring more iterations for a successful attack. This results in adversarial examples that are less perceptible to human vision, enhancing their stealthiness.

Interestingly, by appropriately utilizing these optimizers to minimize the loss, we can leverage adversarial attacks for encryption purposes. Specifically, we can manipulate images so that they are classified correctly only by a specific model. This approach allows us to push an image toward a specific label, creating a form of encrypted communication where only the intended model can decode the image accurately.

Furthermore, by employing image-captioning models, we can achieve even more sophisticated encryption. Instead of pushing the image toward a single text label, we can manipulate it to produce a specific description. This method is more advanced compared to traditional computer vision classification models and can provide a richer and more secure form of encrypted information. This approach opens up new possibilities in secure communications and data protection, where adversarial techniques are used not to deceive but to encode information that only authorized systems can interpret.
