# Efficiency considerations

## Batch size efficiency

It's easy to get a rough swag at batch size efficiency (impact on total samples
per second) by sweeping; even on my CPU machine we can take good advantage of
batch to increase sample rate.

This is a measurement of some state of the NN architecture (subject to change)
with fake inputs; therefore, input preprocessing time is not considered, just
JAX:XLA:CPU time.

```
$ for i in 8 16 32 64 128 256 512 1024; do python attempt.py --batch-size $i --time-step-only; done
bs8        fwd time approximately:     692 us (34.27%)
bs8        step time approximately:  2,019 us; 495.3 steps/s; 3962.4 samples/s
bs16       fwd time approximately:   1,571 us (80.77%)
bs16       step time approximately:  1,945 us; 514.1 steps/s; 8226.2 samples/s
bs32       fwd time approximately:   1,070 us (49.74%)
bs32       step time approximately:  2,151 us; 464.9 steps/s; 14876.8 samples/s
bs64       fwd time approximately:   1,536 us (60.35%)
bs64       step time approximately:  2,545 us; 392.9 steps/s; 25147.3 samples/s
bs128      fwd time approximately:   1,474 us (46.35%)
bs128      step time approximately:  3,180 us; 314.5 steps/s; 40251.6 samples/s
bs256      fwd time approximately:   2,348 us (43.59%)
bs256      step time approximately:  5,386 us; 185.7 steps/s; 47530.6 samples/s
bs512      fwd time approximately:   3,817 us (38.96%)
bs512      step time approximately:  9,796 us; 102.1 steps/s; 52266.2 samples/s
bs1024     fwd time approximately:   5,828 us (34.44%)
bs1024     step time approximately: 16,920 us; 59.1 steps/s; 60520.1 samples/s
```

Plotting this out, 256 is a nice point around the knee of the curve. This says
nothing about learning efficiency, of course, just machine efficiency. The
corpus seems to have millions of examples so hopefully a larger minibatch size
will be tolerable.
