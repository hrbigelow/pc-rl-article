$
\newcommand\grad[2]{\tfrac{\partial #1}{\partial #2}}
\newcommand\ggrad[2]{\dfrac{\partial #1}{\partial #2}}
$

# Introduction

The following analyzes a thinking, learning agent moving about in the world as a pair of interacting systems - a brain, and the *brain's environment* (that is, the agent's body, and the agent's environment).  Each system interacts with the other along an interface.  In this case, the interface is the thin boundary of flesh surrounding the brain.  The two systems produce output, and receive input, in reciprocal fashion.  Each system updates itself according to its own state plus the inputs it receives.  The following diagram illustrates this reciprocal, dynamic relationship.

<div style='display: flex; flex-direction: row;'>
<img alt="" width=400 style='flex: 1 1 1; align-self: center;' src="images/world_loop1.svg">
<img alt="" width=400 style='flex: 1 1 1; align-self: center;' src="images/unrolled1.svg">
</div>

$n$ represents the set of all neurons in the brain.  For simplicity we assume that each neuron takes on some 'firing state' at time $t$, where $t$ takes on integer values.  Each neuron computes its next state from the states of all of its parents and synaptic strengths (weights) $w_t$.  For simplicity of notation, let $n$ and $n_t$ represent both the function and the value of the function at time $t$, respectively.

$s$ is the firing state of the sensory axons traversing the interface.  It represents a tiny subset of the information in $k$.  The functional relationship could be thought of as a matrix multiplication $s = Mk$ where $M$ is a set of one-hot row vectors.  Similarly, $m$ is the firing state of the motor axons traversing the interface.  It too represents a tiny subset of the total brain firing state $n$.  The functional relationship can also be represented with a one-hot matrix.  These 'selector' functions are just placeholders to make the calculations more understandable.  Temporally, they are instantaneous, because, for example, $m_t$ is just a subset of $n_t$, it is not computed from $n_{t-1}$.

So, we have:

$$
\begin{aligned}
n(\cdot) & \equiv n(n_{t-1}, s_{t-1}, w_{t-1}) \\
& = n(n(n_{t-2}, s_{t-2}, w_{t-2}), s_{t-1}, w_{t-1}) \cdots & \mbox{update depends on own state plus sensory inputs} \\
n_t & \equiv \mbox{value of $n(\cdot)$ at time $t$} \\
\grad{n_t}{s_{t-1}} & \equiv \mbox{partial of $n(\cdot)$ w.r.t. $s_{t-1}$ evaluated at time $t$} \\
k(\cdot) & \equiv k(k_{t-1}, m_{t-1}) = k(k(k_{t-2}, m_{t-2}), m_{t-1}) \, \cdots & \mbox{update depends on own state plus motor outputs of brain} \\
k_t & \equiv \mbox{value of $k(\cdot)$ at time $t$} \\
\grad{k_t}{m_{t-1}} & \equiv \mbox{partial of $k(\cdot)$ w.r.t. $m_{t-1}$ evaluated at time $t$}
\end{aligned}
$$

Finally, $o$ represents an objective function whose gradients define how weights will be updated.  Like $n$ and $k$, I use the notations $o(\cdot)$, $o_t$, and $\grad{o_t}{n_t}$.  I also assume it is an instantaneous calculation.

The dashed connections indicate the functional relationship (and state $k$) is unknown to the brain.  It only knows the subset $s$.





Combining the non-brain body and the organism's environment together into $k$ has a few interesting consequences.  First, note that the body and the organism's environment both influence each other.  For example, when you throw a baseball, the baseball accelerates due to muscular action, but the weight of the baseball affects how the arm accelerates in proportion to the amount of muscular force.  It is not possible to learn to throw a baseball without the baseball because the relationship to the motor commands $m$ and the resulting proprioceptive commands $s$ depends both on the weight of the arm, the strength of the muscles, and the weight of the baseball.

Note that this objective function merely defines how weights are updated.  It does *not* say anything about the RL notions of reward, or policy.  If after being trained, weights are frozen, the agent will still be capable of goal-directed behavior.  But, such goals will have been implicitly learned as a consequence of the original objective function and training.


# Learning from actions

For the organism to learn from its actions, it must be able to compute the gradient $\tfrac{o_t}{m_{t-2}}$, following the red path as shown below.  There are just two timesteps, because the $o \rightarrow n$ segment and $s \rightarrow k$ segments are instantaneous as mentioned above.

<div style='display: flex; flex-direction: row;'>
<img alt="" width=400 style='flex: 1 1 1; align-self: center;' src="images/world_loop2.svg">
<img alt="" width=400 style='flex: 1 1 1; align-self: center;' src="images/unrolled2.svg">
</div>


A first attempt would be:

$
\begin{aligned}
\grad{o_t}{m_{t-2}} & = \grad{o_t}{n_t} \grad{n_t}{s_{t-1}} \grad{s_{t-1}}{k_{t-1}} \grad{k_{t-1}}{m_{t-2}}
\end{aligned}
$

However, the last two terms are not computable.  They are asking for a gradient of an unknown function $\grad{}{m_{t-2}} s(k(k_{t-2}, m_{t-2}))$.  Not only is the function $k(\cdot)$ unknown, but it depends on an unknown state $k_{t-2}$.  One way the brain could compute this gradient would be to approximate the function and then compute the gradient of that.  Given that the function itself is recursive, and that the brain calculations are recurrent, it would make sense to try to approximate such a function recursively as well.  Indeed, the brain will have experienced the sequence of $(s_1, m_1), (s_2, m_2), \cdots, (s_{t-1}, m_{t-1})$.   It could incorporate this information into some state variable to help better predict $s_{t+1}$ from $m_t$.

Naturally the only place to encode all of this history is in the current firings $n_t$ and weights $w_t$, so we arrive at the notion of a *reconstruction* of the sensory input, $\hat{s}(n)$.




<div style='display: flex; flex-direction: row;'>
<img alt="" width=400 style='flex: 1 1 1; align-self: center;' src="images/world_loop3.svg">
<img alt="" width=400 style='flex: 1 1 1; align-self: center;' src="images/unrolled3.svg">
</div>


So, let's define the *reconstruction* of $s$, using additional top-down parameters $w^{rec}$, trained by a new *reconstruction objective* $r$, whose gradients are shown in green.

$$
\begin{aligned}
\hat{s}_t & \equiv \hat{s}(n_{t-1}, w^{rec}_{t-1}) \\
s_t & \equiv s(k(k_{t-1}, m_{t-1})) \\
\grad{\hat{s}_t}{m_{t-1}} & = M \grad{\hat{s}_t}{n_{t-1}} \\
\end{aligned}
$$

If we assume that the reconstruction approximates well over all time, then its gradient should also approximate the true gradient well:
$\hat{s}_t \approx s_t \, \forall t 
\implies \grad{\hat{s}_t}{m_{t-1}} \approx \grad{s_t}{m_{t-1}}$

Comparing the previous derivation, we have_:

$$
\begin{aligned}
\ggrad{o_t}{m_{t-2}} & = \ggrad{o_t}{n_t} \ggrad{n_t}{s_{t-1}} \ggrad{s_{t-1}}{k_{t-1}} \ggrad{k_{t-1}}{m_{t-2}} & \mbox{four red arrows} \\[1.5em]
& \approx \ggrad{o_t}{n_t} \ggrad{n_t}{s_{t-1}} \ggrad{\hat{s}_{t-1}}{n_{t-2}} \ggrad{n_{t-2}}{m_{t-2}} & \mbox{two red, then two green arrows}
\end{aligned}
$$

The first expression traces the gradient through the unknown function, such that the last two terms are not computable.  The approximate expression shares the same first two terms, but the second are the last two green arrows.

The reconstruction is similar to a decoder in the autoencoder setup, however, there is a temporal lag.  The reconstruction is predicting one time step into the future.  This is where the notion of 'predictive coding' comes from.

# Conclusions

## Learning from actions



I make two conclusions from this.  The first one is that, in order to implement a credit assignment "back in time" across many timesteps, a brain must necessarily implement an approximation to its environment function $m \rightarrow s$.  This approximation exists in autoencoders and in the theory of predictive coding.

The second conclusion is that organisms don't make "choices" in any 'free will' sense.  The brain's motor output is in no way any different of a computational mechanism than any other part of the brain.  But, the reality of this no-free-will has no bearing on the capability of an organism to "learn to take better actions".  Mathematically, this learning is simply a consequence of the gradient of some objective that flows backward in a loop.



