from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import qiskit


def main():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    state = Statevector.from_instruction(circuit)
    probabilities = {
        str(bitstring): float(probability)
        for bitstring, probability in state.probabilities_dict().items()
    }

    assert abs(probabilities.get("00", 0) - 0.5) < 1e-9
    assert abs(probabilities.get("11", 0) - 0.5) < 1e-9

    print(f"Qiskit funziona! Versione: {qiskit.__version__}")
    print(circuit)
    print("Probabilita:", probabilities)


if __name__ == "__main__":
    main()
