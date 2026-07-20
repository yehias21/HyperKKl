# Enhanced HyperKKL_dyn

This branch contains an improved version of the HyperKKL dynamic variant together with the
evaluation harness used to measure it. Everything below was found empirically over roughly
300 controlled experiments on the Duffing and Van der Pol systems, with every reported
change confirmed by a paired comparison at two random seeds.

## Headline result

On Duffing, with all changes applied, the input conditioned observer beats the autonomous
observer for the first time.

| seed | dyn | autonomous | ratio |
|------|-----|------------|-------|
| 42 | 0.1313 | 0.1606 | 0.817 |
| 43 | 0.1445 | 0.1671 | 0.865 |

Metric is mean steady state RMSE over four input regimes and both in distribution and out
of distribution signal parameters. Lower is better. A ratio below one means the conditioned
observer is better than the same observer with no input conditioning. The original code
gives a ratio of about 1.20, so the conditioned path previously performed worse than doing
nothing at all.

Run to run variation also improved substantially. On the original configuration the dyn
metric varies by up to 94 percent across random seeds while the autonomous metric varies by
about 1 percent. With the changes here the dyn seed spread falls to roughly 10 percent.

The gap to a tuned Extended Kalman Filter narrowed from about 8.2x to about 2.2x on Duffing.

## Correctness fixes

These are defects in the original pipeline. Each was verified numerically against a high
accuracy reference before being changed.

**Phase 1 latent labels were first order accurate, not fourth order.** In `_vectorized_z_sim`
the forcing term was held constant across all four Runge Kutta stages, which destroys the
order of the method. Measured against a DOP853 reference at tolerance 1e-11 the labels
carried up to 2.8 percent relative error, and halving the step size halved the error rather
than reducing it sixteen fold. The forcing is now evaluated at the correct stage times.

**Training targets and the runtime observer used different forcing conventions.** Data
generation built the latent targets using the right endpoint of each step while the observer
rollout propagated the latent using the left endpoint. The measured discrepancy was 5.6
percent and it was a systematic bias rather than noise. Both now use the same convention.

**The input window was offset by one sample between training and inference.** Training built
the window as u[t minus omega, t minus 1] while inference built it as u[t minus omega plus
1, t]. Verified with an index encoded ramp signal. The two conventions are now aligned.

**The dynamic variant was evaluated through a different code path than the one it was
trained for.** `simulate_observer` dispatched on whether the hypernetwork exposes a `step`
method, which is true for both hypernetworks, so the windowed branch was unreachable. Every
evaluation therefore ran the hypernetwork one step at a time with recurrent state carried
across the whole trajectory, while training always called it on a fresh 100 step window.
Inference now uses the same windowed interface as training. Because the weight deltas depend
only on the input window, and the latent ODE does not depend on the deltas, the trajectory
can be decoded in one batched pass. This is numerically identical to the sequential loop to
2.2e-8 and about 36 times faster.

**The low rank weight deltas were not low rank.** `_generate_deltas` emitted the delta in
shape (d_in, d_out) while `nn.Linear` stores weights as (d_out, d_in), and the modulation
helper reshapes rather than transposes. On any non square layer the applied perturbation was
therefore a reshape of the transpose, which is full rank and unrelated to the intended rank r
matrix. The layout is now correct. Note that this fix scores worse in isolation and better
once the other changes are present, which is why it survived only because correctness was
treated as independent of the metric.

**The first layer of each map had no activation.** In `KKLNetwork.forward` the first linear
layer was applied without a nonlinearity and the second immediately followed with one, so the
two composed into a single linear map and one full hidden layer did no work.

## Method changes

**Complex conjugate spectrum for the latent dynamics.** The standard choice
`A = -diag(1, 2, ..., n_z)` makes every latent coordinate a low pass filter of the same
scalar output, so the coordinates are correlated by construction. The measured condition
number of the latent covariance is 6.8e7. Replacing the real diagonal with two by two
complex conjugate blocks decorrelates the latent and improves both accuracy and
reproducibility. A three point sweep of the imaginary part at fixed real part locates an
interior optimum at a pole angle of 45 degrees, that is a damping ratio of 0.707. A separate
sweep shows the overall bandwidth is a stronger axis than the pole angle, and that the usable
bandwidth is bounded by the integration step through the Runge Kutta stability limit, which
gives the practical rule that the product of the largest eigenvalue magnitude and the step
size should stay below about 2.785.

**Bounded and fast saturating activation on the decoder.** The decoder is the only network in
the inference path, since the autonomous observer and the hypernetwork both use the inverse
map alone. Comparing activations on the decoder gives tanh 0.355, hardtanh 0.399, SiLU 0.410,
ReLU 0.421 and softsign 0.433 on the autonomous metric. Smoothness alone buys little and
boundedness alone buys little, but the two together are strongly superadditive, and the
distinguishing property of tanh is how quickly it saturates within the range the decoder
actually operates in. Note that softsign is bounded and 1 Lipschitz like tanh yet performs
worse than ReLU, so this is not a Lipschitz constant effect.

**Physics residual enforced across the input range.** The residual was evaluated at zero
input only, which is a measure zero slice of the operating envelope. The measured residual
is affine in the input with a slope such that at unit forcing it is about 23 times its value
at zero, so the entire forcing dependent part of the residual was left for the hypernetwork
to absorb. The residual is now evaluated over a range of inputs controlled by
`phase1.pde_u_scale`. Important caveat: this helps Duffing with an interior optimum near 1.5
and monotonically harms Van der Pol, where the correct setting is zero. It must be tuned per
system and should not be treated as a general improvement.

**Penalty on the magnitude of the decoder weight deltas.** Instrumentation shows the squared
delta magnitude grows to about 8.9 during training while contributing nothing to either loss
term. Adding a penalty shrinks it by a factor of about 65 and improves the phase 2 training
loss as well as validation, which indicates the large deltas were an optimisation pathology
rather than useful capacity.

## Evaluation harness

`autoresearch/experiment.py` is the harness that produced every number above. It trains phase
1 and the requested phase 2 method and scores the result under a fixed protocol with fixed
initial conditions, fixed signal parameters and a fixed settle time. It reports steady state
RMSE rather than SMAPE, because SMAPE is unstable near the zero crossings that these
oscillatory systems spend most of their time near. It also reports the autonomous observer
scored on the identical protocol, which is the control that matters, and it can compute an
Extended Kalman Filter reference using the same model knowledge that the physics loss
requires.

Two caveats on the harness. The development and test splits share one of eight initial
conditions because the underlying Latin hypercube sampler places points on a fixed lattice,
so the two splits differ mainly in their signal parameters rather than fully in their initial
conditions. And the phase 1 cache keys on a hash of every source file that can affect phase
1, so it is safe, but it must be cleared if you change anything outside that set.

## What did not work

Recording these because they cost real time and the negative results are informative.

Suppressing the decoder Jacobian through a penalty hurts monotonically, from 0.344 with no
penalty to 0.479 and then 0.684 as the penalty rises. The decoder must remain steep in order
to invert a nearly degenerate latent, so the Lipschitz constant is not an independently
controllable quantity and reducing it simply trades one error term for another.

Reducing the latent dimension from five to three improves the latent conditioning by four
orders of magnitude and still performs worse, because it also removes capacity. Conditioning
and dimension are separate axes and only the spectrum route helps.

Enriching the training input distribution with additional waveform families does not help. A
twenty configuration sweep adding chirp, pseudo random binary, multisine, sawtooth, ramp and
filtered noise signals found that only a step input helped at all, and then only marginally,
while the broadband families were the worst. Separately, the trained model generalises
without difficulty to input families it never saw, and is in fact strongest on discontinuous
broadband inputs. The training input distribution appears to be saturated already.

Applying a smooth activation to the encoder rather than the decoder does not help, because
the encoder does not appear in the inference path at all.

## Reproducing

```
conda env create -f kkl.yml
conda activate kkl
pip install "smt==2.6.3"

python -m autoresearch.experiment --systems duffing --method lora --ekf
python -m autoresearch.experiment --systems duffing --method lora --seed 43
```

Note that `smt` must be pinned, because version 2.14 removed the `random_state` option that
`src/systems.py` passes to the Latin hypercube sampler.

## Status

The sub parity result is established on Duffing at two seeds. It does not currently extend
to Van der Pol, where the correct setting for the physics residual range differs and the
conditioned observer still loses to the autonomous one. Treat the Duffing result as
established and the cross system claim as open.
