## Q&A

### Resources

- Deep RL course: http://rll.berkeley.edu/deeprlcourse/
- David Silver's tutorial on RL: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.htm
- Reinforcement learning Video: https://www.youtube.com/watch?v=ggqnxyjaKe4
- RL: https://lilianweng.github.io/lil-log/
- RL: https://spinningup.openai.com/en/latest/
- Bandits book: https://tor-lattimore.com/downloads/book/book.pdf
- David Silver's slides on Deep RL: http://icml.cc/2016/tutorials/deep_rl_tutorial.pdf
- David Silver's video on Deep RL:
http://videolectures.net/rldm2015_silver_reinforcement_learning/
- Sutton's book on RL:
https://web.stanford.edu/class/psych209/Readings/SuttonBartolPRLBook2ndEd.pdf

---

- **some examples**

  - self-driving car _(aws deepracer, tesla)_
  - datacenter cooling
  - buying selling stock
  - nlp, alignment provides example to model that what are safe or unsafe

- **meaningful project idea**

  - an application

- **experience relay and deep Q learning**

  - data comes from complete trajectory
  - at this process, unfold many state every consecutive state is a function
  - instead of collecting data, immediately updating models the data resides in experience relay buffer and then we sample from that buffer
  - this is collected across many trajectories and thats one attempt to reduce the correlation between successive states

- **how to choose neural architecture for the deep-Q learning ?**

  - neural network arch. depends on the form of the input, if we applying for the gameplay agent then the input is some signal or screen, but the part of network that does the processing of data
  - vision also we can use transformer model, what input the network is collecting and depending on that we can choose our network. 
  - and once we have powerful features

- **difference in between bandits methods and RL**

  - next state depends on previous where previous hold the information like state, reward the action etc.
  - now the bandits are simplified version of RLs where we drop some of the featuers
  - for every item I have some metadata, what category of product what's the title of movie, genre, director and customer that information.
  - here in contextual bandits, again the arms can be one of item, each movie and arm there can be there have a feature.
  - actions are very shortsighted, how an action have the impact in state transition, In RL, we have those states

- **what is discount factor**

  - give us leverage the trade-off immediate reward or future reward we get
  - put it very low, maximize the reward for the current state
  - kind of also weighing future reward as good as the current reward are
  - the other reason to need discount factor for non-episodic task
  - in those cases helps us to assign finite value
  - more for mathmatical reason, without discount factor the definition of value function makes no sense

- **- What is UCB, and Explain the intuition and application of it? How confidence interval limit calculated?**

  - one of bandits algorithm
  - so the I mean it's class of algorithm
  - we maybe pulling some suboptimum action
  - it will let you pull each arm some no. of times
  - for every arm we have select it and give reward to arms
  - gives us idea for the average of all arms.
  - how can we factor in the confidence, 
  - it computes confidence interval and look at what is maximum reward we can collect if to select that action.
  - **[resource](https://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/)**
  - how often we selected arm in past other factor is what magnitude of reward we have seen

- **If RL is not reliable for future prediction then**

  - wrong statement, inaccuracies comes due to several different other reasons
  - paid-off between sample complexity, how risky decision the model can come, before it actually learns the Off policy learning,
  - can make many suboptimal decision, cost of making these decision very high
  - each decision can be very expensive
  - for Off policy, it takes actually longer to converge but it tends to be more

- **Integrate RL in transformer**

  - can use transformer also
  - search .. search ..

- **Multiple methods optimized for your recommendation**
  - AV Tests, cohort of users randomly split into control trafic and other, treatment subject to new model
  - can compare between then groups

- **evaluate peformance model in sequential learning setup**

  - for ml objective, trained on very specific loss function,
  - Q learning, very squared loss function used to provide more data points, just like supervised model how the loss function in training data detect for overfitting and convergence.
  - this explore and exploit not sequential decision, completely new to system, don't have any information should explore and find out.
  - context depended
  - exploration budget should be, reduced overtime, system static, gradually reduce the amount of exploration, automatic, nothing else changes the estimation go more robust overtime, model will do exploitation now

-
  - I have let's say n product to recommend so the context, it actually combines the information about the user and information coming from the items
  - the all info we know is to capture the context

<br>

- **concept of shortterm and longterm reward**

  - doesn't really connected with when it is reward
  - observer reward from the environment
  - customer may not immediately buy, she may added the card, she will comeback and make the purchase
  - long term reward where we don't have any delay in feedback, whereas short term reward
  - long term reward that we accumulate that policy from the record.

- **how to mue0 and trimmer handles persability in environment ?**
  - where we don't observer complete state but partially observer it, slightly more involved then the MDP
  - **[resource]()**

**-- Lunch Break ----**

**3:25 Continue**

- **Difference in between RL and SL**

  - In RL, we take decision those have effect in environments, SL on other hand we trying to predict the next word there
  - there's no interaction between environment and agent here
  - In RL, we more forward for the action and reward which continuous

- **do actor-critic network uses both dq and ...  ?**

  - we want to learn those action that only give us the ...
  - precisely captures the high value function
  - loss function assigning loss for policy network
  - weightage being applied that where's the value network picking from.
  - waydown the iteration of that sample

- **exploration and exploitation in RL**

  - don't have any need data to start with so collect the data to start that's what comes in exploration
  - not explore at all
  - most of the technique on space
  - randomly taking some action or sampling action, guided by some other principles
  - exploitation resonably confident what decision take

- How do you handle the issue of sparse rewards in reinforcement learning tasks?

  - https://openreview.net/forum?id=IvJj3CvjqHC
  - https://proceedings.neurips.cc/paper/2019/file/16105fb9cc614fc29e1bda00dab60d41-Paper.pdf
  
- **policy and policy gradient**

- **issues of cold start in RL**

- **how do RL incorporte environment modelling ?**

  - then in a limited sector like a customer simulator for certain domain which modelled by physics, then there's well grounded simulator
  - main challenge is that they are very domain specific quite a hard problem to solve
  - you have information that when customer is simulating

- **Q-Learning vs R-Learning**

  - R-Learning

- **what are the advantages and limitation of policy gradients techniques over value-based ?**

  - both are popular, In real difference, so I think biggest will be if very large action space
  - most of recent neither policy or value based so more like hybrid
  - complementry, state of the art algorithm, combine policy based and value based technique
  - 

- **key challenges to train RL agents in adversial environments**

  - multi-agent setup, each agent wants to maximize its own reward and it wins over other agent
  - defining the reward function, not just reward, also reward that's given to other agent
  - different decision choices that problem setup 
  - mmm .. 

- **Upper confidence bound**

  - confidence internval with lower and upper value
  - UCB is just upper value
  - at what freq, algo keep on selecting suboptimal arm
  - details are hidden in the UCB algorithm, at the core

- **thompson sampling**

  - another algo in bandit setup, go over the ppt for this
  - idea is that you assume its a parametric setup, we don't know the parameter
  - reward is sample from the output
  - starts with some idea that what parameter could be
  - then some evidences sample from the distribution, so all that prior can be combined in our best knowledge estimate called posterior distribution
  - and know imagine posterior distribution have very high variance
  - not very certain about the reward
  - end up selecting certain choice sort of away from the mean value
  - likely to go in either side of mine
  - variance of posterior distribution 

- 