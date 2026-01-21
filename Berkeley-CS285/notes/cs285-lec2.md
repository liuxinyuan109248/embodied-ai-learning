# CS285 Lecture 2: Supervised Learning of Behaviors

## Terminology & Notation in Sequential Decision Making

### Core Variables

The fundamental components of the agent-environment interaction are defined as follows:

- **$s_t$ (State):** The underlying configuration of the system at time $t$.
- **$o_t$ (Observation):** What the agent actually perceives (e.g., a camera image of a tiger).
- **$a_t$ (Action):** The decision or movement made by the agent based on its observation (e.g., choosing a path to run away).

### The Policy ($\pi_\theta$)

The policy represents the "brain" of the agent, typically implemented as a neural network (shown in the diagram as a CNN):

- **$\pi_\theta(a_t | o_t)$:** A partially observed policy, where the action is conditioned on the current observation.
- **$\pi_\theta(a_t | s_t)$:** A fully observed policy, used when the agent has perfect information about the state.

### Probabilistic Graphical Model

The bottom diagram illustrates how these variables interact over time:

- **State Transitions:** The next state $s_{t+1}$ depends on the current state $s_t$ and action $a_t$, represented by the transition probability $p(s_{t+1} | s_t, a_t)$.
- **Observation Mapping:** The state $s_t$ produces an observation $o_t$.
- **The Markov Property:** The model assumes that the future ($s_{t+1}$) is independent of the past ($s_{t-1}$) given the present state ($s_t$).

### Visual Workflow

- **Input:** The agent receives an observation $o_t$ (Image).
- **Processing:** The policy $\pi_\theta$ (Neural Network) processes the observation.
- **Output:** The agent executes an action $a_t$ (Decision).

## Challenges and Solutions in Imitation Learning

### The Core Limitation of Behavioral Cloning (BC)

Although Behavioral Cloning is the simplest form of Imitation Learning, it is **not guaranteed to work** in real-world sequential environments.

- **Key Distinction:** It differs from standard Supervised Learning because the **i.i.d. assumption (Independent and Identically Distributed) does not hold.**
- **The Problem of Compounding Errors:** In a trajectory, a single small mistake by the agent leads to a state slightly outside the training distribution. Since the agent hasn't seen this "off-track" state before, it makes a larger mistake, eventually leading to catastrophic failure.

### Strategies to Address the Distribution Shift

To fix the issue of compounding errors, we can use the following approaches:

- **Smart Data Collection & Augmentation:** Train the model on "recovery" data. For example, using side-view cameras in autonomous driving to teach the model how to steer back to the center.
- **Powerful Models:** Use architectures like **RT-1 (Robotics Transformer)** or Large Language Models (LLMs) that make fewer mistakes initially, thereby slowing down the error accumulation.
- **Multi-task Learning:** Learning multiple related tasks simultaneously to help the model learn more robust and generalizable features.
- **DAgger (Dataset Aggregation):**
  - Train a policy from expert data.
  - Run the policy to collect new trajectories (where the model might fail).
  - Ask the expert to label those new trajectories with the correct actions.
  - Aggregate the data and retrain.

Success in Imitation Learning isn't just about mimicking the expert's perfect path; it's about knowing how to recover when the agent inevitably drifts away from that path.

## Theoretical Analysis of Behavioral Cloning Failure

### The Distributional Shift Problem

In Behavioral Cloning, the fundamental issue is that the agent’s actions at test time affect its future observations, breaking the i.i.d. assumption.

- **Training Objective:** We minimize error under the expert's state distribution $p_{\text{train}}(s)$:
    $$\max_{\theta} \mathbb{E}_{s \sim p_{\text{train}}(s)} [\log \pi_{\theta}(a|s)]$$
- **Testing Execution:** When the policy $\pi_{\theta}$ is actually run, the states are sampled from its own resulting distribution $p_{\pi_{\theta}}(s)$.
- **The Gap:** Because $\pi_{\theta}$ is not perfect, $p_{\pi_{\theta}}(s) \neq p_{\text{train}}(s)$. A small error $a_t \neq \pi^*(s_t)$ leads to a new state $s_{t+1}$ that drifts away from the training data.

### Error Accumulation Analysis ($O(T^2)$)

We can formalize the "compounding error" by analyzing the expected cost over a time horizon $T$.

**Definitions:**

- **Cost function:** $c(s, a) = 0$ if $a = \pi^*(s)$ (matches expert), and $1$ otherwise.
- **Bounded single-step error:** Assume the probability of making a mistake on the training distribution is bounded by $\epsilon$:
    $$\mathbb{E}_{s \sim p_{\text{train}}} [\pi_{\theta}(a \neq \pi^*(s)|s)] \leq \epsilon$$

**The Worst-Case Derivation:**

1. **State Distribution Decomposition:** At any time $t$, the state distribution $p_\theta(s_t)$ can be viewed as a mixture of being "on-track" or "off-track":
   $$p_\theta(s_t) = (1 - \epsilon)^t p_{\text{train}}(s_t) + (1 - (1 - \epsilon)^t) p_{\text{mistake}}(s_t)$$

   - $(1 - \epsilon)^t$: The probability the agent has made **zero mistakes** up to time $t$.
   - $(1 - (1 - \epsilon)^t)$: The probability the agent has made **at least one mistake** and entered an unknown distribution $p_{\text{mistake}}$.

2. **Distributional Divergence:** Using the useful identity $(1 - \epsilon)^t \geq 1 - \epsilon t$ for $\epsilon \in [0, 1]$, we can bound the distance between our current distribution and the training distribution:
   $$\left| p_\theta(s_t) - p_{\text{train}}(s_t) \right| = (1 - (1 - \epsilon)^t) \left| p_{\text{mistake}}(s_t) - p_{\text{train}}(s_t) \right| \leq 2\epsilon t$$
   *(Note: The factor of 2 comes from the maximum total variation distance between two distributions)*.

3. **Total Expected Cost over Horizon $T$:**
   We sum the expected cost at each time step $t$:
   $$\sum_{t=1}^T \mathbb{E}_{p_\theta(s_t)}[c_t] \leq \sum_{t=1}^T \left( \mathbb{E}_{p_{\text{train}}(s_t)}[c_t(s_t)] + \left| p_\theta(s_t) - p_{\text{train}}(s_t) \right| c_{\max} \right)$$
   Since $\mathbb{E}_{p_{\text{train}}}[c] \leq \epsilon$ and the divergence is $\leq 2\epsilon t$:
   $$\text{Total Cost} \leq \sum_{t=1}^T (\epsilon + 2\epsilon t) \in O(\epsilon T^2)$$

### The "Perfect Expert" Paradox

The analysis reveals a counter-intuitive reality in robotics:

- **The Paradox:** If an expert never makes mistakes, the training data $\mathcal{D}_{\text{train}}$ only contains "perfect" states.
- **The Consequence:** The agent never learns how to recover because it never sees states that are "slightly off".
- **The Solution:** Imitation learning is often more robust if the training data includes **recoveries**. This can be achieved via data augmentation (e.g., side-view cameras for cars) or interactive algorithms like **DAgger**.

Here is the continued technical note in English Markdown, covering the slides on Non-Markovian behavior, Multimodal distributions, and Diffusion policies.

## Failure Modes and Advanced Solutions in Imitation Learning

### 1. Non-Markovian Behavior

A common reason Behavioral Cloning fails to fit the expert is the assumption that behavior is **Markovian**: $\pi_\theta(a_t | o_t)$.

- **The Conflict:** Standard policies depend only on the current observation. However, human demonstrators are often **Non-Markovian**; their actions depend on the entire history of observations: $\pi_\theta(a_t | o_{1:t})$.
- **The Solution: Sequence Modeling:** To utilize the whole history, we can use architectures that integrate temporal information:
  - **LSTMs / RNNs:** Maintaining a hidden state to aggregate past information.
  - **Transformers:** Using self-attention mechanisms over a sliding window of past observations.
- **Causal Confusion:** Simply adding history can sometimes backfire. A model might focus on "nuisance variables" (e.g., a brake indicator light) rather than the actual cause of an action (e.g., a pedestrian), leading to poor generalization when those correlations break.

### 2. Multimodal Behavior

The second major failure mode is **Multimodality**. In many scenarios, there is more than one "correct" way to perform a task (e.g., a car can go left or right around an obstacle).

- **The Problem with Regression:** If we use a standard Mean Squared Error (MSE) loss, the model will attempt to average the modes.
- If Mode A is $-1$ and Mode B is $+1$, the average is $0$.
- In a driving scenario, "averaging" a left turn and a right turn results in driving straight into the obstacle.

#### Solution A: More Expressive Continuous Distributions

To represent multiple peaks in a probability distribution, we can use:

1. **Mixture of Gaussians (MoG):**
The policy outputs weights $w_i$, means $\mu_i$, and variances $\Sigma_i$ for multiple Gaussian components:
$$\pi_\theta(a_t | o_t) = \sum_{i=1}^K w_i \mathcal{N}(a_t; \mu_i, \Sigma_i)$$

2. **Latent Variable Models (e.g., Conditional VAEs):**
The policy takes a latent noise variable $\xi$ as input along with the observation. By sampling different $\xi$, the model can produce different valid actions from the same observation:
$$a_t = f_\theta(o_t, \xi), \quad \xi \sim \mathcal{N}(0, I)$$

#### Solution B: Diffusion Policies

Diffusion models represent the state-of-the-art for handling high-dimensional, multimodal action spaces. Instead of predicting the action in one shot, the model learns to iteratively refine a noisy signal into a clean action.

- **Forward Process:** Gradually add noise to the true action $a_{t,0}$ until it becomes pure Gaussian noise $a_{t,T}$.
- **Reverse Process (Inference):** The learned network $f_\theta(s_t,a_{t,i})$ predicts the noise or the gradient to step from a noisy action $a_{t,i}$ back toward a realistic action $a_{t,i-1}$.
  $$a_{t,i-1} = a_{t,i} - f_\theta(s_t, a_{t,i})$$
- **Benefit:** This approach is extremely expressive and can represent very complex, non-parametric distributions of behavior, leading to much smoother and more robust robot control.

### 3. Discretization of Action Spaces

An alternative to complex continuous distributions (like Mixtures of Gaussians) is to discretize the action space. By converting continuous control into a classification problem over discrete bins, the model naturally handles **multimodality** (e.g., assigning high probability to two different bins representing "left" and "right").

#### The Dimensionality Challenge

While discretization is effective for 1D actions, it becomes impractical in higher dimensions due to the "curse of dimensionality." If we have an action vector $a \in \mathbb{R}^d$ and we discretize each dimension into $B$ bins, the full joint space contains $B^d$ possible discrete actions—an exponential growth that makes training a single softmax classifier impossible.

#### The Solution: Autoregressive Discretization

To handle high-dimensional actions without the exponential cost, we discretize **one dimension at a time**. By applying the **chain rule of probability**, we can decompose the high-dimensional joint distribution into a sequence of conditional 1D distributions:
$$p(a_t | s_t) = p(a_{t,0} | s_t) \cdot p(a_{t,1} | s_t, a_{t,0}) \cdot p(a_{t,2} | s_t, a_{t,0}, a_{t,1}) \cdots p(a_{t,d-1} | s_t, a_{t,0}, \ldots, a_{t,d-2})$$

**Implementation Details:**

- **Sequential Modeling:** This structure is typically implemented using sequence models like **LSTMs** or **Transformers**.
- **Iterative Prediction:**
  - The network takes the state $s_t$ (and potentially previous action components) to predict the distribution for the first dimension $a_{t,0}$.
  - The sampled value of $a_{t,0}$ is fed back into the model to predict the distribution for the next dimension $a_{t,1}$.
  - This continues until all $d$ dimensions of the action vector $a_t$ are sampled.

**Key Advantages:**

- **Efficiency:** Instead of $B^d$ outputs, the model only needs to output $d \times B$ values.
- **Dependency Modeling:** It captures the correlations between different action dimensions (e.g., the required steering angle might depend on the chosen acceleration).
- **Flexibility:** It retains the ability of discrete classifiers to represent arbitrary, non-Gaussian distributions.

## Goal-Conditioned Behavioral Cloning (GCBC)

A major extension of Imitation Learning is moving from a single-task policy to a multi-task, goal-oriented policy. This helps address the limitations of basic Behavioral Cloning by teaching the agent the relationship between states, actions, and desired outcomes.

### 1. The Multi-Task Advantage

- **Single Task:** A standard policy $\pi_\theta(a|s)$ learns to reach a specific point $p_1$. If the starting position or environment changes significantly, the policy may fail because it only knows one specific behavior.
- **Multi-Task:** A goal-conditioned policy $\pi_\theta(a|s, p)$ learns to reach *any* point $p$. By training on multiple goals ($p_1, p_2, p_3$), the model learns a more robust, generalized mapping of "how to move toward a target" rather than just memorizing a single path.

### 2. Formalism and Distributional Shift

In Goal-Conditioned Behavioral Cloning, we treat a successful demonstration $\{s_1, a_1, \dots, s_T\}$ as a sequence of actions intended to reach the final state $s_T$.

- **Training Objective:** For each demonstration $i$, we maximize the likelihood of the actions given the current state and the ultimate goal (the final state of that trajectory):
  $$\max_{\theta} \sum_{i} \sum_{t} \log \pi_\theta(a_t^i | s_t^i, g = s_T^i)$$
  
- **Two Types of Distributional Shift:**
  - **State Shift:** The agent encounters a state $s_t$ during test time that was not in the training data.
  - **Goal Shift:** The agent is given a goal $g$ (e.g., a specific image or coordinate) that was never seen as a terminal state during training.

### 3. Learning Latent Plans from Play

Data collection for robotics is often expensive. A powerful solution is to use **"Play Data"**—non-structured, teleoperated sequences where a human simply "plays" with the environment without a specific task in mind.

- **The Architecture (LMP):**
  - **Plan Recognition:** An encoder takes an entire sequence and compresses it into a latent plan $z$.
  - **Plan Proposal:** At test time, a module proposes a latent plan $z$ based on the current state and the desired goal image.
  - **Goal-Conditioned Policy:** The policy $\pi_\theta(a | s, g, z)$ executes actions based on the state, the goal, and the latent plan.
- **Benefit:** This allows the robot to learn complex behaviors (like opening drawers or picking up objects) from raw, unlabelled interaction data.

## Beyond Pure Imitation: Iterated Supervised Learning

Pure imitation learning is limited by the quality and quantity of expert demonstrations. We can go beyond this by using **Iterated Supervised Learning** (e.g., algorithms like GCSL), which allows an agent to learn from its own experience.

### The Self-Improvement Loop

The core idea is to treat the agent's own past (even failed) attempts as training data by "re-labeling" them.

- **Start with a Random Policy:** The agent begins with no knowledge and acts randomly.
- **Collect Data with Random Goals:** The agent attempts to reach various goals. It might fail to reach the intended goal, but it will inevitably end up at *some* state $s_{final}$.
- **Hindsight Relabeling:** Treat the actual reached state $s_{final}$ as the "intended" goal for that trajectory.
  - Even if the agent failed its original task, that trajectory is a "perfect demonstration" of how to get to wherever it actually ended up.
- **Improve the Policy:** Perform behavioral cloning on this relabeled data:
    $$\max_{\theta} \mathbb{E}_{(s, a, g) \sim \mathcal{D}} [\log \pi_\theta(a|s, g)]$$
- **Repeat:** As the policy improves, the agent reaches more distant and complex goals, creating a flywheel of self-improvement.

## Addressing Distribution Shift: DAgger

As established, the primary failure mode of Behavioral Cloning is that the agent enters states not covered by the expert training distribution, leading to compounding errors. **DAgger (Dataset Aggregation)** is a meta-algorithm designed to solve this by forcing the training distribution to match the agent's actual execution distribution.

### 1. The Core Intuition

- **The Problem:** In standard BC, the training trajectory distribution $p_{\text{data}}(o_t)$ differs from the agent's expected trajectory distribution $p_{\pi_\theta}(o_t)$.
- **The Solution:** Instead of trying to make the agent's distribution match the expert's, we collect training data from $p_{\pi_\theta}(o_t)$. We run the agent's current policy to see where it goes, and then ask an expert to tell the agent what it *should* have done in those specific states.

### 2. The DAgger Algorithm

The algorithm follows an iterative loop to aggregate data:

1. **Initial Training:** Train a policy $\pi_\theta(a_t | o_t)$ using an initial human-collected dataset $\mathcal{D} = \{o_1, a_1, \dots, o_N, a_N\}$.
2. **Interaction:** Run the current policy $\pi_\theta(a_t | o_t)$ in the environment to generate a new dataset of observations $\mathcal{D}_\pi = \{o_1, \dots, o_M\}$.
3. **Expert Labeling:** Ask a human (the expert) to provide the correct actions $a_t$ for every observation in $\mathcal{D}_\pi$.
4. **Aggregation:** Merge the new labeled data with the existing dataset: $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_\pi$.
5. **Iteration:** Repeat the process until the policy converges or performance is satisfactory.

### 3. Practical Challenges of DAgger

While DAgger provides strong theoretical guarantees for reducing the error bound from $O(T^2)$ to $O(T)$, it faces significant practical hurdles:

- **Step 3 (The Human Bottleneck):** The most problematic step is requiring a human to manually label every frame collected by the robot.
- **High Cognitive Load:** It is often counter-intuitive for humans to provide a precise numerical action (like a steering angle) just by looking at a static image $o_t$ without actually being in the "flow" of driving.
- **Safety Concerns:** Running an unoptimized, "half-trained" policy $\pi_\theta$ in a real-world environment (like a physical car) to collect $\mathcal{D}_\pi$ can be dangerous if the agent performs unpredictable or hazardous actions.
