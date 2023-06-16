#initialization
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

# importing Qiskit
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# import basic plot tools and circuits
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT

def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

def inverse_qft(circuit, n):
    """Does the inverse QFT on the first n qubits in circuit"""
    # First we create a QFT circuit of the correct size:
    qft_circ = qft(QuantumCircuit(n), n)
    # Then we take the inverse of this circuit
    invqft_circ = qft_circ.inverse()
    # And add it to the first n qubits in our existing circuit
    circuit.append(invqft_circ, circuit.qubits[:n])
    return circuit.decompose() # .decompose() allows us to see the individual gates

def estimate(phi,n):
    qpe = QuantumCircuit(n+1,n)
    qpe.x(n)
    for qubit in range(n):
        qpe.h(qubit)
    repetitions = 1
    for counting_qubit in range(n):
        for i in range(repetitions):
            qpe.crz(4*np.pi*phi, counting_qubit, n)  # controlled-RZ
        repetitions *= 2
    qpe.barrier()
    # Apply inverse QFT
    qpe = inverse_qft(qpe, n)
    #qpe = qpe.compose(QFT(n, inverse=True), [i for i in range(n)])
    # Measure
    qpe.barrier()
    for k in range(n):
        qpe.measure(k, k)
    qpe.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
    plt.show()
    aer_sim = Aer.get_backend('aer_simulator')
    shots = 2048
    t_qpe = transpile(qpe, aer_sim)
    results = aer_sim.run(t_qpe, shots=shots).result()
    answer = results.get_counts()
    plot_histogram(answer)
    plt.show()
    max_number = max(answer, key=answer.get)
    final_phi = int(max_number,2) / 2**n
    return final_phi


print(estimate(1/11,5))

