# Physics reference

This page documents the physics stand-ins: the 1D transfer-matrix boost
simulator, the antenna-coupling model and the scalar objectives. They are
qualitatively faithful to MADMAX behaviour and analytically verified where
possible, but they are **placeholders** — the calibration algorithm only
touches them through {class}`~madmax_calibration.simulator.BoostSimulator`
(`beta2`, `predict_J`) so the real MADMAX simulation can be substituted
behind the same interface (see {doc}`hardware`).

## The 1D dielectric-haloscope model

A dielectric haloscope boosts the axion-induced electromagnetic signal
with a perfect mirror followed by $N$ dielectric disks separated by vacuum
gaps. In one dimension, in each homogeneous region $r$ with refractive
index $n_r$, the axion field sources a constant particular solution

$$
E_{a,r} = -\frac{E_0}{n_r^2},
$$

(with $E_0 \equiv 1$ in our units) and the total field is

$$
E_r(x) = A_r e^{i k_r (x - x_r)} + B_r e^{-i k_r (x - x_r)} + E_{a,r},
\qquad k_r = \frac{2\pi \nu\, n_r}{c}.
$$

Boundary conditions close the system:

- **Mirror** (perfect conductor at $x = 0$): total field vanishes,
  $A_0 + B_0 + E_{a,0} = 0$.
- **Interfaces**: continuity of $E$ and $\partial_x E$ between adjacent
  regions. The jump in $E_{a}$ across an index change is what injects the
  axion source into the propagating solution.
- **Radiation condition**: in the semi-infinite vacuum region on the
  antenna side there is no incoming wave, $B_{R-1} = 0$.

This yields one banded linear system of size $2R-1$ per frequency
($R = 2N + 2$ regions), which
{func}`~madmax_calibration.simulator._beta2_curves` solves *batched over
the frequency grid* with a single `numpy.linalg.solve` call. The **boost
factor** is the outgoing intensity relative to the mirror-only emission:

$$
\beta^2(\nu) = \left| A_{R-1}(\nu) \right|^2 .
$$

Dielectric loss enters through a complex index
$n_d = n\,(1 - \tfrac{i}{2}\tan\delta \cdot e^{\theta_{\log loss}})$.

### Analytic checks

Two closed-form results anchor the implementation (both are enforced by
`tests/test_simulator.py`):

- **Mirror only**: $\beta^2 = 1$ at all frequencies.
- **Transparent mode** (half-wave gaps, half-wave disks): amplitudes add
  coherently,

  $$
  \beta = 1 + 2N\Bigl(1 - \frac{1}{n^2}\Bigr),
  $$

  e.g. $\beta^2 = 45.6976$ for $N = 3$, $n = 5$ — matched by the solver to
  $10^{-6}$.

### Sensitivity scales

With the prototype 3-disk, $n = 5$ stack at 21 GHz ($\lambda/2 \approx
7.1$ mm gaps), the objective responds to geometry errors at the
several-hundred-µm scale (sensitivity grows toward the upper end of the
18–24 GHz range as the gaps shrink):

| Error | Effect on $J_{\text{scan}}$ |
|---|---|
| stack offset $+0.8$ mm, gap compression $+0.4$ mm | ≈ −40% |
| same magnitudes, opposite compression sign | ≈ −7% |
| stack offset $+0.6$ mm, gap compression $+0.3$ mm | ≈ −23% |
| loss scale $e^{1}$ | few % |

Note that error *combinations* matter: some $(\theta_z, \theta_c)$ pairs
partially compensate each other in the stack response — this produces the
near-degenerate inference directions discussed in {doc}`statistics`.

## Detector-state parameters $\theta$

{class}`~madmax_calibration.simulator.DetectorState` carries the three
inferred nuisance parameters and their controllability labels
(parent proposal §5):

| Parameter | Geometry effect | Correctable? | Compensating control |
|---|---|---|---|
| `z_offset` | whole stack displaced from the mirror | yes | $z_{\mathrm{global}}$ |
| `compression` | disk $i$ displaced by $i \cdot c$ (uniform inter-disk gap error) | yes | disk mode 0 + $z_{\mathrm{global}}$ |
| `log_loss` | scales the dielectric loss tangent | **no** | — (diagnostic-only) |

The cancellation algebra (verified in
`test_control_basis_cancels_correctable_errors`): $\theta$ displaces disk
$i$ by $\theta_z + \theta_c\, i$; disk mode 0 displaces disk $i$ by
$a_0 (i+1)$ and $z_{\mathrm{global}}$ by $z_g$; hence

$$
a_0 = -\theta_c, \qquad z_g = -\theta_z + \theta_c
$$

cancels the correctable errors exactly.

## Control basis

{func}`~madmax_calibration.control.disk_mode_basis` builds the
low-dimensional disk-correction basis $B$ (columns = displacement of disk
$i$ per unit mode amplitude):

- **mode 0** — $\propto (i+1)$: uniform change of *all* gaps,
- **mode 1** — $\propto (i+1)^2/N$: linear gradient in the gaps,
- **mode 2** — $\propto (i+1)^3/N^2$: quadratic gap profile (off by default).

$q_{\mathrm{disk}} = q_{0,\mathrm{disk}} + B a_{\mathrm{disk}}$ with
$q_0$ from the (stand-in) offline optimization
{func}`~madmax_calibration.simulator.nominal_half_wave_geometry` /
{func}`~madmax_calibration.simulator.optimize_nominal_gaps`.

## Antenna coupling

The receiver coupling is modelled as a separable efficiency
$\eta \in (0, 1]$ ({meth}`~madmax_calibration.simulator.BoostSimulator.coupling`):

$$
\eta(u_A, z_f)
= \exp\!\Bigl(-\frac{\lVert u_A - u_{A}^{\mathrm{beam}}\rVert^2}{2 w^2}\Bigr)
\cdot \frac{1}{1 + \kappa_f\,(z_f - z_f^{\mathrm{opt}})^2},
$$

with beam width $w = 4$ mm and focus curvature $\kappa_f = 2 \times 10^5$
m⁻². The *simulator* assumes a centred beam and $z_f^{\mathrm{opt}} = 0$;
the *real* (mock) detector has offsets in both. Consequences:

- the transverse beam offset is found and corrected online by Step 3
  (antenna alignment) — it never needs to enter the inference;
- the focus-optimum offset is invisible to the simulator and appears as a
  learnable *discrepancy* over $z_{\mathrm{focus}}$ — exactly the
  confounding structure the Step-5 model is designed to absorb honestly.

The measured curve is $\beta^2_{\mathrm{meas}}(\nu) = \eta \cdot
\beta^2(\nu)$; the antenna-aligned prediction used by Steps 1/5/6 is
{meth}`~madmax_calibration.simulator.BoostSimulator.predict_J` (beam term
$= 1$, focus term retained — the "plug-in antenna optimum" convention of
Step 6 design §5).

## Scalar objectives

The parent proposal (§6) requires $J$ to be a physics figure of merit that
respects the MADMAX **area-law trade-off**: for a fixed disk system, peak
boost trades against bandwidth, so "maximize the peak" and "maximize the
band" are physically different calibrations. Three objectives are
implemented in {mod}`madmax_calibration.objectives`:

`scan_rate` (default)
: $J = 10^{-4}\,\langle \beta^4 \rangle_W$ — axion scan rate scales with
  $\beta^4$, so this is the natural rate proxy over the window.

`smooth_min`
: $J = -\tau \log \langle e^{-\beta^2/\tau} \rangle_W$ — soft minimum over
  the window; a broadband-robustness objective.

`peak`
: $J = \max_W \beta^2$ — narrow-band confirmation objective.

### Curve → objective uncertainty

{meth}`~madmax_calibration.objectives.Objective.with_uncertainty`
propagates the per-bin curve uncertainty to $\sigma_J$ by Monte Carlo
(Step 4 design §9.5), splitting the uncertainty into a **shared**
(normalization-like, fully correlated across the band) and an
**independent** per-bin component (§9.4). For the $\beta^4$-type objective
this correctly yields roughly twice the relative curve error.
