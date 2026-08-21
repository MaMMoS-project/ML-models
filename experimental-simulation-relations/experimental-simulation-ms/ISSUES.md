This ML project has a *SERIOUS* issue and should not be used for production:

During the validation project it turned out that there is a misleading flag in the
Materials Project database: "theoretical" does *NOT* mean simulated vs. experimental
value but rather "exists in theory" vs. "is known to exist from experiments".

Thus essentially in this project there is NO experimentally measured data and hence
no valid experiment-vs-simulation error correction can be found by ML

The development of this model has been stopped until this issue is solved
