## Example
```
% crf_learn -t template train.data model                                               [~/work/sequence-labeling/kudo]
 CRF++: Yet Another CRF Tool Kit
 Copyright (C) 2005-2013 Taku Kudo, All rights reserved.

 reading training data:
 Done!0.00 s

 Number of sentences: 3
 Number of features:  54
 Number of thread(s): 8
 Freq:                1
 eta:                 0.00010
 C:                   1.00000
 shrinking size:      20
 iter=0 terr=0.63636 serr=1.00000 act=54 obj=24.16947 diff=1.00000
 iter=1 terr=0.13636 serr=1.00000 act=54 obj=19.32509 diff=0.20043
 iter=2 terr=0.13636 serr=1.00000 act=54 obj=16.02792 diff=0.17062
 iter=3 terr=0.00000 serr=0.00000 act=54 obj=15.47175 diff=0.03470
 iter=4 terr=0.00000 serr=0.00000 act=54 obj=15.43538 diff=0.00235
 iter=5 terr=0.00000 serr=0.00000 act=54 obj=15.33969 diff=0.00620
 iter=6 terr=0.00000 serr=0.00000 act=54 obj=15.33837 diff=0.00009
 iter=7 terr=0.00000 serr=0.00000 act=54 obj=15.33726 diff=0.00007
 iter=8 terr=0.00000 serr=0.00000 act=54 obj=15.33722 diff=0.00000

 Done!0.01 s

 % crf_test -v2 -m model test.data                                                      [~/work/sequence-labeling/kudo]
 -- INSERT --
 jojonki@jmbp () % crf_test -v2 -m model test.data                                                      [~/work/sequence-labeling/kudo]
# 0.032005
Sara    NP      I-PER   I-PER/0.396177  I-LOC/0.341696  I-PER/0.396177  O/0.262127
is      NP      O       O/0.615654      I-LOC/0.246562  I-PER/0.137784  O/0.615654
going   NV      O       O/0.762365      I-LOC/0.118818  I-PER/0.118818  O/0.762365
to      NP      O       O/0.680411      I-LOC/0.202291  I-PER/0.117298  O/0.680411
Tokyo   NP      I-LOC   I-LOC/0.609739  I-LOC/0.609739  I-PER/0.158437  O/0.231824
in      NP      O       O/0.680411      I-LOC/0.202291  I-PER/0.117298  O/0.680411
California      NP      I-LOC   I-LOC/0.609739  I-LOC/0.609739  I-PER/0.158437  O/0.231824

```
