## Basic

At the very early stage of ADMM_IPC research, we performed unit test (with only one point and one segment). To pass this test, we found the following
 keys in ADMM pipeline:

- Use xTilde to initialize ADMM update variables
- Process with global()-local()-dual() order

Test data could be found [linking0](https://paper.dropbox.com/doc/ADMM-IPC-Basic--A_tzxUFoMCjAX3AySu3mU~qdAg-bp6r0xf9V2ieV59IB611V) and
[linking1](https://paper.dropbox.com/doc/Unit-Test-Edge-Triangle-for-ADMM-Framework--A_vCYgXEJ63tXidI~lNzxQ9kAg-SlPk6BiQ4bSPYuVuLrj3u). We also have some 
[theoretical analysis](https://paper.dropbox.com/doc/ADMM-IPC-Analysis-theoretical--A_sL~2Ut3EvkTrRVcSrGZNEqAg-rNtoAgCSBDMF8dGD1aMHa), thanks to Danny and Minchen.

## Formulating ADMM update variables

To achieve Translation-free, we need to make sure local solve results is translation-free as well, so that collision point wouldn't be fixed at any point in
 the space. We could reduce ADMMupdate variables DoF via substract the first vertex position from all the remainings.

We also need to have the correct classification for different types of collision. General classification should be applied here as PE-only for 2D and PT/EE for 3D. 
There are some [comparision](https://paper.dropbox.com/doc/ADMM_GIPC--A_uEKufMxeeiTlWkayarnIBUAg-zOEYHTj1MGcZIKXXmO7A8) between different classification ways.

The results for this could be found [linking2](https://paper.dropbox.com/doc/ADMM-IPC-Translation-free--A_vllaYR~AwFxAGq__TIOf24Ag-bkM3SzKonGMoBsqHJbNRD)

## Penetration-free

To make sure there is no penetration, we adopt the following strategies:

- Global filter. This includes CCD after obtaining global linear solve resutls and extra segment-triangle intersection check.
- Local filter. 

More details could be found [linking3](https://paper.dropbox.com/doc/ADMM-IPC-Penetration-free--A_v5uvVQSgacE~3IjcWP5DC4Ag-gT9wDcgsiUZm4tafVQw11)


## Q weighting

We observed some very weird behaviors with bad Q settings. The artifacts couldn't be reproduced during the later experiments. Some results are listed [linking4](https://paper.dropbox.com/doc/ADMM-IPC-Constant-Q-Weighting--A_up636BkUz4XLp2r2H2xD3oAg-oigTcWWATnVyLKga3kSVL)

We also found some [evidences](https://paper.dropbox.com/doc/Latest-ADMM-Results--A_v1x~LvQSytXWpbQfuedqX0Ag-CtIwZOPdL0qXQF3hezjoB#:uid=397624691933570595160799&h2=Tunnel-is-a-good-example-which) that clamping Q could help convergence. But these evidences are not strong and are based on few examples which may be biased.

## Current Issues

With normal stiffness, ADMM IPC performs well. We now find that ADMM IPC have these issues:

- sticky when youngs modulus is small
- motion damping when youngs modulus is large

We do have 3D results both for [ADMM IPC](https://paper.dropbox.com/doc/Latest-ADMM-Results--A_v1x~LvQSytXWpbQfuedqX0Ag-CtIwZOPdL0qXQF3hezjoB) and [NEWTON IPC](https://paper.dropbox.com/doc/Newton-IPC--A_u_k9e2Oe_rT9Qm9jbmmk66Ag-k6K7lAJvkORTvg2Sm8jV4) (with [timing](https://paper.dropbox.com/doc/Benchmark-Timing--A_v3Al~lT0gLvLTdwoPiWEiIAg-TZLqIiHQVAyeevGTr1AjE))

