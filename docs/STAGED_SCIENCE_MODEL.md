# Scientific model and staged acceptance — 2026-11-22

This updates the accepted repair plan in `CODE_REVIEW_2026-11-20.md` using the
user's experience with nuclear dipoles and GP absorption of point-source flux.
It is a model specification and work order, not a claim that these extensions
are implemented. The implementation baseline is commit
`531612b2ccb3f25ed44db2ee1478ed2461715050`.

## Observations that the design must explain

1. Nuclear residuals show dipoles. Candidate causes include a subpixel centroid
   error, relative astrometric error, PRF/pixel-phase mismatch, and resolved or
   asymmetric nuclear emission. None is established by the residual appearance.
2. The flexible galaxy GP absorbs field-star flux; adding point sources did not
   by itself produce satisfactory residuals. A longer GP scale competes with the
   need to model the peaked nucleus. The same ambiguity threatens transient flux.
3. Work to date concentrated on ch2. Cross-band physical consistency is a later
   objective, not a justification for imposing an untested fixed color now.
4. Flux is nonnegative. Share transient flux across BCDs with a total span at
   most one day; split and flag candidate spans over one and under two days.
   Do not use single-linkage chains to extend the window. Background offsets
   may be signed because they correct calibrated/subtracted measurements.
5. There is no external reference transient light curve. Internal consistency,
   independent injected truth and held-out exposures provide the validation.

## Proposed generative model

For band b, exposure i and native pixel p:

    y[b,i,p] = B[b,i,p] + F[b,i](H[b] + N[b])[p]
               + sum_s a[b,s] P[b,i,p](x_s)
               + f[b,e(i)] P[b,i,p](x_T) + epsilon[b,i,p].

- `H`: smooth diffuse galaxy brightness, regularized by a Matérn GP. Start with
  one spatial scale expressed in angular or native-pixel units with an explicit
  conversion to scene pixels. Its scale and amplitude are learned primarily
  from template data, with source components included. Sweep scales rather than
  adopting a larger kernel without evidence.
- `N`: compact, static nuclear emission. Compare nested candidates: shifted
  unresolved source; resolved elliptical Gaussian; unresolved core plus resolved
  envelope. Add more flexible profiles only when held-out data require them.
  When N is unresolved, evaluate it with P rather than a scene-grid delta.
- `a`: explicit field-star fluxes, initially shared across exposures within a
  band. Check this assumption; flag variable stars instead of forcing their
  variability into the galaxy or detector calibration.
- `f`: nonnegative transient flux per one-day epoch, exactly zero in verified
  transient-free templates. Use a common sky position, with a calibrated position
  uncertainty. Template contamination is a failure mode to test, not assume away.
- `B`: start with per-BCD constant backgrounds. Add gradients only when held-out
  residuals justify them, with tests of degeneracy with diffuse host emission.
- `F`: the established native forward operator for extended surface brightness.
- `P`: detector-integrated, phase-dependent point-source response, with full
  detector coordinates and band-appropriate calibration. Do not integrate a PRF
  twice, normalize each cropped stamp to unity, or tune it on the transient.

Physical source amplitudes and total modeled astrophysical brightness must be
nonnegative. A signed GP perturbation around a positive mean is a mathematical
parameterization, not permission for negative physical emission. Specify where
positivity is enforced and whether it makes inference nonlinear. Do not quietly
replace the GP covariance with independent pixels for a large problem.

Keep detector nuisance parameters distinct from sky morphology. Estimate exposure
registration from field stars with an explicit reference-frame gauge. Start with
translation; add rotation/distortion only when constrained. The nucleus and
transient have separate sky centroids. Never recenter the entire scene to remove a
nuclear dipole without checking the field stars and the transient position.

For resolved N, forward-model an intrinsic extended profile through the instrument
response. A blur applied to native pixels is not automatically equivalent. Test the
zero-width limit against the native point renderer, and verify flux and aperture
conventions. Profile complexity is selected on templates and held-out exposures,
not by the recovered transient flux or visual cleanliness alone.

## Identifiability and leakage controls

Templates establish the static host, nucleus and field stars. Learn structural
choices and hyperparameter priors there, then carry their uncertainty into science
inference. A plug-in fixed host can understate flux errors; blindly freeing a
flexible host again on science pixels can absorb the transient. Implement a
conditional/joint likelihood or an equivalent uncertainty-propagating method and
prove equivalence on small cases. Do not count template likelihood twice when
reusing a template posterior.

Explicit source columns alone do not establish identifiability. Measure flux
transfer among GP, nucleus, stars and transient, posterior correlations, objective
curvature/near-null modes and sensitivity to kernel scale. Do not force the GP to
be orthogonal to point-source columns as a default: that is a modeling constraint
which can change real host emission and must earn its use through validation.

Required comparisons: GP only versus GP + stars; GP + point nucleus versus GP +
resolved nucleus; correctly centered versus offset sources; longer versus shorter
GP scale; template-rich versus template-poor cases; transient-free versus science
joint fits. Artificial stars and transients go on backgrounds with different
morphology from the fitted family. Include a source near the nucleus and a
transient indistinguishable from a static component when templates are absent.
The latter should report non-identifiability, not manufacture a precise flux.

Use recovery slopes across many injected fluxes/seeds to estimate attenuation.
Report component leakage, bias in Jy and fractional bias where truth is nonzero,
RMSE, coverage, failure rates, and sensitivity to masks, rolls and dithers. A
single passing pull or visually smooth residual is not an acceptance test.

## Noise and residual interpretation

Use BCD uncertainty maps and document their units, gain/exposure assumptions and
existing Poisson/read-noise terms. Do not estimate noise with sqrt(Jy) or add a
Poisson term already included in the uncertainty product. Compare signed residual
maps, fitted centroid-derivative modes, radial/tangential structure, ACF/lag
correlations, phase/roll trends, and signed dipole versus centroid.

Training residuals are reduced and correlated by fitting. Their normalized width
need not be one. Use held-out BCD predictions with measurement plus predictive
uncertainty, including relevant covariance, and posterior-predictive/simulation
checks for training residuals. Distinguish pointwise predictive variance from a
full whitening operation. Propagate registration and GP/nuclear uncertainty.

At zero flux, constrained estimates are biased upward and ordinary symmetric
Gaussian intervals have wrong coverage in general. Compare validated profile-
likelihood, simulation-calibrated or posterior intervals; state which probability
interpretation is claimed. Report nondetections/upper limits without allowing
negative physical flux or clipping an unconstrained error bar into a fake interval.

Residuals consistent in amplitude with measurement noise may still contain
systematic structure. Quantify it with multiplicity-aware diagnostics across
pixels/frames and an injection bias check. Never increase noise floors simply to
make chi-square look acceptable.

## Stages and promotion gates

0. Publish the existing repair baseline as a reviewable PR. No new science claims.
1. Build deterministic fixtures, reference diagnostics, and paired performance CI.
2. Validate isolated point sources, real PRF normalization, pixel phase and
   registration, without a galaxy GP. Both bands may have renderer unit tests;
   scientific fitting remains ch2 initially.
3. Fit template-only nucleus candidates, increasing complexity only when held-out
   performance improves. Include neighboring stars and a controlled smooth host.
4. Introduce the diffuse GP and explicit stars/nucleus; quantify component leakage
   and establish template-derived priors. Select complexity using independent
   validation partitions, reserving a final untouched test set.
5. Implement a scalable exact-objective solver, preserving the reference solutions
   and calibrated operators. Interface design can follow Stage 1; scientific
   equivalence must be retested against Stage 4 before promotion.
6. Infer transient flux with host/nucleus uncertainty, one-day grouping, positivity,
   zero-flux coverage, and blinded injection/held-out tests.
7. Only after single-band gates pass, add optional cross-band sharing. Share sky
   geometry first; allow independent flux/color. Then test a positive-semidefinite
   multiband GP covariance with independent per-band residual components. The
   zero-correlation limit must reproduce independent fits. Account for different
   PRFs, masks, coverage and nonsimultaneous epochs. Never force achromatic host
   morphology or constant transient color without evidence.

At each stage run one BCD, multiple dithers in one epoch, multiple rolls/templates,
and only then the full supported data volume. Keep fixed sky footprint comparisons
across SR=1,2,4. Record exactly which data and calibration files are used and their
checksums. An unavailable real-data fixture is an explicit unmet gate, not a
synthetic pass renamed as real validation.

The user has not supplied a fractional science-bias tolerance. Report measured
bias and confidence intervals, expose tolerances as declared configuration, and
keep the scientific acceptance gate unapproved until a tolerance is agreed. Do
not select the threshold after observing a result. Numerical identity tolerances
and mechanical CI performance budgets can be specified independently.

## Efficiency requirements

Use the small exact dense implementation as a reference, never as a requirement
for production memory scaling. Preserve cross terms, constraints, adjoints and
nuisance-marginalized uncertainty in scalable methods. Log convergence residuals,
KKT checks, objective, failures, peak memory, setup/solve/predict time, operator
calls and calibration reads. No silent diagonal-Hessian shortcut or GP fallback.

Performance CI runs base and PR code on the same runner, fixture and thread count,
in isolated processes, with cold and warm-cache cases reported separately. Warm
up numerical kernels; use repeated runs and medians/dispersion. Correctness gates
precede speed comparisons. Suggested initial mechanical budgets are >25% wall-time
regression (and >0.25 s absolute) or >20% peak-memory growth (and >20 MiB absolute),
with a confirming rerun; calibrate these budgets on runner variability and record
any intentional change in the PR. A noisy result is inconclusive, not a pass.
These are proposed engineering budgets, not measured baseline tolerances.

Required PR CI: small operator/adjoint/flux/constraint regressions, small leakage
and injection tests, deterministic operation-count/allocation bounds, and paired
performance checks. Extended manual CI: larger coverage experiments, real-PRF
checks, held-out real BCD diagnostics, and scaling curves. Cache invalidation,
input immutability, process cleanup and bounded test timeouts are also requirements.
Save numerical JSON/NPZ, seeds/config/hash manifests, JUnit, and labeled plots.
Do not put large/private BCD products in Git or download them in untrusted fork CI.

## Sources and status of recommendations

The component separation, stages and gate design above are proposed engineering
and inference choices to test, not assertions that a particular nuclear profile
or kernel is physically correct.

- [IRAC Handbook: pixel phase](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/59/)
  describes PRF centering and native-interval sampling; centroid definitions must
  be consistent between data and PRF.
- [IRAC PRF calibration](https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/calibrationfiles/psfprf/index.shtml)
  documents the supplied products and normalization. Audit the actual local files
  and mission phase before interpreting their amplitudes as calibrated flux.
