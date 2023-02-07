# TensorSimulations
Codebase to learn Simulating Quantum Circuits with Tensornetworks. The library
that is being used is `tensornetwork` maintained by Google. It is reccommended
that you use `quimb` since it is more actively maintained.


## Dummy
Dir. contains rough code for trying matrix multiplication and gate operation

## Gate
Dir. contains code to apply gates to a density matrix.

## Kraus
Dir. contains code to apply Kraus operators to density matrix.

### Depolarization
Submodule of Kraus that applies the Kraus operators for depolarization noise.
The Kraus operators are constructed with the help of google cirq.

## [TODO] Superoperator
Dir. contains code to apply superoperators operators to density matrix.

#TODO:
    - Compare the performance for Kraus and superoperator

