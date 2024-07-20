## Q&A

**Backlog** : Revise previous topics and learn more about sequencial learning

---

- *HMM is part of speech from which we predict the sequence of word*
  - ex. **Play me Arabian Nights**
  - do we have enough data
  - probability updation and emission probabilities *( tough one )*

- **What are the best practices for setting hyperparameters in transformer models?**
  
  There are various strategies involved. But the most important parameters are usually the number of layers, size of each layer, learning rate, regularization related parameters like dropout. We should try out with first startign out with a simpler model and then increase the complexity gradually

- *We are trying to move toward transformer based solution, but it's not all that work for all kind of problems, there's some solution that need RNNs, transformer typical take longer time, so totally depend on the application, but more or less companies moving toward LLM and*

- **Resources**
  - on CourseEra
  - Stanford Playlist
  - 

- **Druing HMM lot of matematics, formulas and algorithm ?**
  
  - **Viterbi Algorithm**
    - basically dynamic programming
    - what is the probability that the word is predicted ?
    - probability of the word and state is independent
    - we don't have to actually calculate all the paths
    - SMT, till some time sequence what is the probability of reaching the sequence
    - if you know the previous one, we just need to find for each state for particular timestamp.
    - probability of word fixed irrespective time
    - by knowing these probabilities, we know what the sequence of the hidden terms
    - hidden state depends what we trying to find

  - *Forward and Backward Algorithm*

  - *HMMs : Final computation of parameters*

  - *HMMs : Semi-Supervised Model*

  - *HMMs : Posterior computation*
    - we can caculate new probabilities by using the current alphas,
    - unsupervised and supervised data treated in different ways

- **In what scenarios sequential learning is better than other types of learning methods ?**

  sequential learning is specifically suitable for scenarios where incoming data is in the form of seq such as strem of text, stream of stock values etc. There are many ML techniques that deal with fixed inputs instead of sequences as well like Image classification etc. where other forms of inputs could be there

- **How do we preprocess and prepare sequential data for modeling ?**

  It depends on the domain but lets say we are using sequential models in text, typically we should apply some pre-processing to remove noisy text (like html tags, bad characters etc.) and then we do tokenization of text using sub-word tokenizers

- **What challenges does Alexa face in maintaining context over long conversations, and how are these addressed?**

  We need to increase the context length of models. In modern LLMs like Claude there is support for large(over 200k) context length. ALso in traditional transformers, approaches like longformer help support larger context. Another approach could be to summarize historical text and use that for future predictions

- **different between two types of attention mechanisms**
  - *Bahdanau Attention*
    - context vector provided
    - we use the decoder hidden state in first output and then do some computation
    - additive attention

  - *Luong Attention*

- **what are the disadvantages of Recurrent models, and how LLM solving that ?**

  - RNN suffer from the phenomena called *exploring gradients* and *vanishing gradients*

- **decoder models and how they work ?** *( learn again )*

  - two kinds of layers : *encoders* and *decoders*

  - *encoders* layer encode each word in input and transform into for example in a first word of sentence
  - so for example the sentence is ummm... here is a dog, the word it refers to the dog.

  - similarily *decoders* we do similar task

- **Pretraining thorough language modeling**

  - https://arxiv.org/pdf/1511.01432.pdf
  - foundation of GPT models

- **Subword Modelling**

  - In traditional there's need to tokenization technique
  - treating each word as separate token will lead to explode to the vocab, if *complex morphology*
  - there's always limit to what we can done
  - *learning on vocabulary of parts of words/subwords*
  - each subword treated as token, and we try with converting it to a vector.

- **Hello sir, I want to know what problems occur when HMM is applied in real-world and what are its constraints?**

  As told in lecture, most systems nowadays use neural networks which are discriminative. In HMM, we model the joint probability of observed and hidden sequences and find the maximum probability of hidden sequence. In discriminative we find given the observed sequence prdict the probability of hidden sequence. Real life application is mainly the hidden sequence prediction like NER

- **How do you determine which type of positional encoding to use in Transformer models for different tasks, and what are the criteria for selecting the appropriate encoding method ?**

  It is a design choice like any other hyper-parameter. We need to explore various approaches and choose the best one used for your application. One recent approach that is used several LLMs is Rotatory positional embeddings : https://medium.com/ai-insights-cobet/rotary-positional-embeddings-a-detailed-look-and-comprehensive-understanding-4ff66a874d83

- *How would you implement an LSTM or GRU from scratch using a deep learning framework like TensorFlow or PyTorch?*

  There are some great tutorials that can help with this. Sharing one of them : https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091

- *What is the best method for parallelizing transformer training across multiple devices?*

  There are well known techniques to use multiple devices. Some of them are built-in into frameworks like PyTorch. There are also libraries like deepspeed that help with this : https://www.deepspeed.ai/tutorials/

- *Denote some applications HMMs in Amazon.*

  There are several applications. Speech systems like speech-text systems are one place they are frequently used in industry.

*GPT only does decoding, next word prediction*

- *T5 Model : Text-to-Text transfer Transformer*

  - https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html

- Hidden Markove model are more of a generative models,

- *What are some recent advances and breakthroughs in the field of sequential learning?*

  Some of the recent advances are mainly wrt. LLMs. There are sevaral papers recently on how/why LLMs work. Paper on LLama2: https://arxiv.org/abs/2307.09288 is an intertesting one.

- *how to train transformer on CPU ?*

  just don't they are very huge models, requires lot high computation resources.

- *How do LLMs, LVMs perform self moderation apart from blacklisting or masking target words.*

  Good question. There are lot of ideas on how to provide an engaging response and not just moderate/blacklist on sensitive queries. Techniques like RLHF/SFT are useful in these scenarios.

- *Can you share any industry case studies or projects where sequential learning made a significant impact?*

  Alexa is one case study but in general applications related to recommendations, ad prediction are some applications where they have made a big impact : https://towardsdatascience.com/breaking-down-youtubes-recommendation-algorithm-94aa3aa066c6

- *why can't use the transfomer with the case of alexa to make the response more better ?*

  We do leverage in many Amazon systems for language understanding, classifcation etc.

- *how do LLMs peforms self-attention apart from blacklisting ?*

- *what strategy to make HMM robust to noisy observation ?*

  - can't be predicted as such
  - lot of data we do manually annotated
  - differenet methods robustness based methods
  - lot of the robustness essentially data, if data cleanly labelled, then we can mitigate these problem

- *how to avoid overfitting in transformer model ?*
  
  - amount of flexibility we offer during different level
  - better not to train some of the BERT layers
  - regualrization technique, L2 regularization
  - large scale decoder like GPT models
  - require large server system
  - trained on web-scale data
  - few epochs of training, till training and validation data matching increase
  - kept on training it has been overfitting
  - grooking, in which we find we keep on overtraining and training curve and validation become flat
  - somewhere the weight start to decrease in magnitude
  - overfitting in those case where it help

- **Why BERT model superior to other word2vec ?**

  - In word2vec, vectorial representation of every word.
  - Representation of every word is idependant to training, every word represented by vector
  - techniques like BERT, we try to avoid that In terms of how we train BERT masking, subword modelling
  - also amount of data BERT is trained on
  
google colab is recommendation in terms of training model there, google colab llm finetuning

- *images also sequential data, how we process that ?*

  - not a sequential data, but set of pixel vectors
  - divide the image in certain block
  - images can be treated with sequential data also

  another part of sequential data is *video data*, transfomers are good when we have large batch of unsupervised data

- *Is other Neural Networks effect the peformance of sequential learning algorithms ?*

  - for large amount of data go with transformers
  - for less check RNNs and CNNs

- *Good resources on finetuning LLMs*

  - a paper on llama

- **Can you explain the improvements in Mamba over Transformers.**

  Not an expert at this. But this is one resource that I found : https://thegradient.pub/mamba-explained/

- **Which will give us a better summarizer for terms and condition of insurance policy: BERT or T5?**

  I would say using LLMs like mixtral is better if you dont want to finetune. Several cloud providers like AWS Bedrock give easy way to leverage to use LLMs at minimal cost

- *how quantization works ?*

- **Is Fairness in ML an important research area with huge scope ?**

  Yes, ResponsibleAI/fairness is a great reasearch area especially with the advent of LLMs that are prone to handle free-form queries

labelled data of adversial input is very important, again like they are might be better method of adversial traning and all that, but data is bottom line

- **How much data is required to finetune an llm in terms of no of samples to achieve good results? What is the format of the data required?**

  Amount of data depends on model size. LLAMA paper : https://arxiv.org/pdf/2302.13971 is a good resource to understand more on finetuning

- *What are small language models(slm)?*

  They are essentially smaller versions for language models that contain 1-2 Billion parameters or lesser

- *What optimizations should I perform, If I wish to use LLMs on edge devices like Raspberry pi? What about smaller ML models on microcontrollers like Arduino?*

  I think compression techniques can help. Using small language models can be useful as well. But I dont think LLMs would still be small enough to fit in raspberry pi. Better to use LSTMs and small tranformers models.

- *Practical problem faced with pretrained models*

  - for some tasks it works very well, improve it is difficult part
  - if less amount of data to fine tune model on
  - sentiment calssification
  - diverse data is better than larger amount of data

- *when training data unstructured data and then giving it to LLM for summarize and then send that summarize data to LLM training*

  - not generalize as such

- *how better BERT will be used with summarization ?*

CNN works best with images, feature space of image isn't that high, but in text the feature space could be huge.

- *can these LLMs ever be taught to reason? Or are reasoning models separate*
  
  Yes, do refer to Chain-of-Thought reasoning in LLMs

- *so sir for finetuning we need lablellled data ?*

  Yes, finetuning usually requires labelled data gathered manually or though automatic ways such as clicks in ad prediction setup

- *Does the choice of framework, such as PyTorch or TensorFlow, impact the performance of Large Language Models (LLMs) during their creation or fine-tuning?*

  Typically not. either framework should work. There could be small bugs once in a while in these frameworks that can impact peformance. Since its an active developer community those are usually fixed

- *best method is large model and large data, compute and scale unbeaten at this point*

- *solution depends upon the type of hallucination, for example api classification using LLMs*

- **What strategies are effective for making HMMs robust to noisy observations, and how do you ensure that the model remains accurate in the presence of such noise?**

  The best way to make HMMs robust to noisy observations is to annotate the data. By making sure the data itself is high quality we remove all ambiguity with such noisy observations from the data and make the HMM model robust.

- 