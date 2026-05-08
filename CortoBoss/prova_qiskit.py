from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def main():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    state = Statevector.from_instruction(circuit)

    print("Circuito Qiskit:")
    print(circuit.draw(output="text"))
    print("\nStatevector finale:")
    print(state)
    print("\nProbabilita':")
    print(state.probabilities_dict())


if __name__ == "__main__":
    main()
