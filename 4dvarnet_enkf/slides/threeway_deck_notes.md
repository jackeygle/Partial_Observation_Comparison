# 演讲备注 / Speaker notes

## Slide 1: Three methods on the same observations

The EnKF is here as the baseline and it is in every results comparison, but this deck does not open up how the filter works — that is deliberate, not an omission. Its configuration is the original project's, unmodified. If the mechanism comes up, the short version is: 100 perturbed copies of the state, advanced one frame through a pre-trained surrogate, then pulled toward the observations by a Kalman gain built from the ensemble covariance.

## Slide 2: The pipeline, end to end

One figure for both methods, because the honest description of method 3 is 'the same pipeline plus a branch' and two separate diagrams made it look like more than that.

The solver is not a generic network: it runs 20 steps of gradient descent on J(x), and what is learnt is HOW to step, replacing a hand-tuned step size. The gradient comes from autograd, so no adjoint of Phi had to be derived.

The prior is small — 9,420 of 2,051,542, about 0.5% — because almost everything is the ConvLSTM. Its job is to say what a plausible crowd field looks like, and it is the only route by which information reaches cells no robot saw: the observation term is identically zero there.

The next three slides open the three boxes in turn — the prior, then the head at training time, then the five members at inference.

## Slide 3: Inside method 2 — the prior is where the dynamics live

This is the one box in the previous figure that is not generic machinery, so it is worth opening.

Be precise about what the mask zeroes: exactly one voxel, (kt//2, kh//2, kw//2). It is NOT true that Phi ignores frame t — it reads x(t, h+-1, w+-1) freely. What it cannot read is the single cell it is predicting. The paper's Sec. 3.2 wording is psi(x)(s) does not depend on x(s), where s is a space-time point.

Without the mask Phi could learn the identity, x - Phi(x) would be zero everywhere and the prior term of the cost would carry no information. checks/check_paper_conformance.py measures how much identity nonetheless leaks through the two-scale path (2e-2 to 4e-2 relative) — the pooling and the learned upsampling do give it a route back to frame t, which is a deviation from the paper worth naming rather than hiding.

phi's kernel of 1 is what keeps that constraint intact: a pointwise layer cannot reach across time, so no amount of depth after psi can reintroduce a dependence on frame t.

## Slide 4: Method 3 — the same solver, asked for its uncertainty

Why the head sits outside J(x): sigma is not part of the physical state being assimilated, so letting it into the cost would change what the solver descends. Keeping it after the solve preserves 4DVarNet Sec. 3.4 — the gradient handed to the solver still comes from the cost, computed from observations alone.

The loss box separates the two lines on purpose. The bracket is Lakshminarayanan's Eq. 1 verbatim — that much is theirs. The sg(sigma^2)^beta factor in front is NOT in Eq. 1: it is a per-point weight on the whole loss, with the variance detached so it rescales gradients without changing what sigma^2 is fitted to. beta lives in [0,1]; at 0 the factor is 1 and we are back to Eq. 1 exactly, at 1 every point is weighted equally.

If asked whether the weighting is standard: it is a known technique and the provenance is recorded in losses.py, but I have not read that paper yet, so I am presenting beta as our implementation choice justified by the measurement below rather than citing it. That reading needs doing before any of this goes in the thesis.

Why not beta=0, i.e. Eq. 1 as written: we tried it. sigma^2 ran away toward zero and the loss followed. That run still had a per-channel variance floor, so the collapse landed on the floor and turned the loss into an MSE weighted by the inverse floors [154, 5, 67, 11905] — vx, which carries 58.7% of the error, got the smallest weight and came out +203% worse, dragging full-state RMSE +52%. Killed at epoch 42; see runs/FAILED_ml5_beta0.json. The floor is gone now, so beta=0 would diverge rather than mis-weight, but it would still fail.

On the head's initialisation, if it comes up: the last layer starts at W=0, b=-3, so sigma^2 begins flat at about 0.049 everywhere, i.e. sigma ~0.22, which is roughly the model's own RMSE. Starting slightly over-dispersed is deliberate — the 1/(2 sigma^2) term has a benign gradient when sigma^2 is too large and an explosive one when it is too small.

The extra Phi pass for the residual input costs 9,420 of 2,051,542 parameters, about half a percent. What sigma the head actually ends up producing is on the last appendix slide.

## Slide 5: Method 3 at inference — five members, two variance terms

This slide exists because the split decides whether five members are needed, and since the sigma^2 floor was removed the answer has flipped to yes. The learnt sigma alone now scores 0.65 against the five-member 0.87, and its NLL blows up because one member can claim a sigma near zero with nothing to floor it. The member spread is what rescues the total. While the floor existed the aleatoric-only variant was BETTER calibrated than the ensemble, so this was genuinely open then — it is not any more.

Vocabulary, if it is challenged: 'aleatoric' strictly means irreducible given the measurement process. Sensor noise qualifies; a cell no robot visited does not, since more robots would reduce it. Both nevertheless land in the first term, because a single network reports a large sigma either way. The split of the total variance is exact; the naming is an interpretation.

The seed split (--init-seed varying, --data-seed fixed) was added for this: our original single --seed drove both the weights and the robot routes, so five members would each have seen a different observation realisation. Varying the observation noise between members would confound the epistemic term with observation noise, which is not what Sec. 2.4 asks for.

## Slide 6: Result — reconstruction accuracy

Say the second bullet before the number sinks in, not after. The accuracy gap is a smoother-versus-filter comparison and it is flattered by that; the calibration comparison on the next slide is not, because each method is scored against its own error.

The filter's configuration is the original project's, unmodified — worth saying before it is asked.

Asking for the uncertainty costs 1.8% of accuracy. The deep-ensembles paper reports a median penalty of 5.6% for the same substitution, so this is at the cheap end of what that objective normally costs.

## Slide 7: Appendix — the variational cost, written out

This is the same J as on the framework slide, with the channel sum written out as four explicit terms instead of a sigma over c. Nothing is different, it is just spelled out.

The two residuals are the whole story. dy is the observation misfit and it is multiplied by the mask, so it is identically zero wherever a cell was not observed — which means a blind cell reaches the cost ONLY through dx. That is the mechanism by which the prior fills the blind zone.

On the 200 frames, if it is asked: they are simply added in. The sum runs over every point of the window, 200 x 36 x 12 = 86,400 per channel, and the time axis is treated exactly like the two space axes — there is one scalar J for the whole window, not 200 of them. Dividing by N makes each term a mean square, so J does not grow with the window length.

And if it is asked how one scalar can steer 200 frames: the gradient dJ/dx has the same shape as x, so every (channel, frame, cell) gets its own component. The scalar is only what gets differentiated.

The weights are learned, and the table gives the trained values. They exist because the four channels are on different scales — the next slide is about that.

## Slide 8: Appendix — how the prior's convolution computes one value

The quadruple sum is Conv3d written out, and underneath it is the same sum on real numbers — these used to be two slides and they were saying the same thing twice.

The point is the mask. Exactly one voxel is zeroed, (kt//2, kh//2, kw//2). So the output at (t,h,w) cannot read the input at (t,h,w) — but it reads x(t, h+-1, w+-1) freely, and 26 of the 27 taps are live. Saying 'Phi does not use frame t' is wrong and would not survive a question.

That weaker property is all that is needed: Phi cannot be the identity, so x - Phi(x) is a real prediction error rather than something the network can zero out. The blanked centre had the value 2.2973 and the unmasked weight would have added +0.2210 — one term out of 104, which is what the constraint actually costs.

phi's kernel of 1 is what preserves it: a pointwise layer is a per-voxel matrix multiply, so no depth after psi can reintroduce a dependence on the predicted cell. The next slide follows all three layers at this same voxel.

The padding line matters for the window ends: at t=1 and t=200 the missing temporal neighbour is read as zero.

## Slide 9: Appendix — the whole prior, at one voxel

The previous slide covers one of 32 features of the FIRST of three weight layers of ONE of two branches. This is the rest of it, on the same voxel.

The correction to make out loud if the previous slide left the wrong impression: a branch is not one convolution. It is psi (4->32, 3x3x3, the only layer that reaches across time and space), then two pointwise convs 32->32 and 32->4, with a ReLU after each of the first two. Three weight layers, 4,676 parameters. Phi is two such branches plus the learned upsampling: 9,420.

The ReLU rows are worth a sentence — 13 of the 32 features are zero at this voxel after psi, and 16 after the first pointwise layer. That is normal, not a defect: different features fire in different places.

The pointwise layers being kernel 1 is what keeps the zero-tap guarantee alive. At a fixed voxel they are literally a 32x32 and a 4x32 matrix multiply, so no depth after psi can reintroduce a dependence on the cell being predicted.

The two-branch sum at the bottom is Eq. 10 with numbers in it — the coarse branch contributes +0.6113 on density here against the fine branch's +0.3534, so it is not a small correction.

The fine branch and Phi are both asserted against the module's own forward, so if these rows were wrong the script would not have written the json.

## Slide 10: Appendix — from the gates to the update

The framework slide had 'i, f, o, g = chunk(...)' as one line and never said what the four do. The table is that, and the thing to stress is that each of the four is its own 7,776-term convolution over the same input — they share one weight tensor but they are four separate maps.

The cell update is the only place the solver has memory: f decides how much of the previous step survives, i how much of the new content is written. Then o*tanh(c) is what gets read out, and the read-out is a 64-term sum against one row of out.weight. The row index is c*T + t, because the 800 axis is the four channels and the 200 frames flattened together — which is also why dT is fixed by the weight shape and cannot be changed at inference.

One honest observation if it comes up: the gates are not behaving like soft weights. About 39% of forget-gate values sit below 0.05 or above 0.95, and at this voxel 45 of the 64 channels have at least one railed gate. I picked the loudest contributing channel rather than an unsaturated one because at this voxel there is no unsaturated one.

Every line on both solver slides is asserted against the module's own forward in checks/diag_solver_trace.py — the hand sum, the sigmoid, the cell update and the read-out.

## Slide 11: Appendix — the variance head, step by step

Both of the head's convolutions have kernel 1, so at a fixed voxel each is a plain matrix-vector product: 12 to 32, then 32 to 4. The whole head is 548 parameters and fits on one slide.

The 12 inputs are the three things the head is given: the reconstruction, the mask, and the prior residual — four channels each.

The left table is one hidden unit in full. The two large products are the density residual and the density value, and all four mask products are negative: being observed pushes this unit down. That is the head learning 'observed means certain'.

Then layer 2 sums 32 terms — 16 of them zero, because ReLU killed those units at this voxel — and softplus turns the result into a non-negative excess on top of the per-channel floor. Here the excess is 0.000047 against a floor of 0.003239, so sigma is essentially the sensor limit. The next slide is about how often that happens.
