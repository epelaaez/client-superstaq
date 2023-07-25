import cirq
import networkx as nx

import supermarq


def compute_communication(circuit: cirq.Circuit) -> float:
    """Compute the program communication of the given quantum circuit.

    Program communication = circuit's average qubit degree / degree of a complete graph.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the communication feature for this circuit.
    """
    num_qubits = cirq.num_qubits(circuit)

    graph = nx.Graph()
    for moment in circuit:
        for op in moment:
            if cirq.num_qubits(op) == 2:
                graph.add_edge(op.qubits[0], op.qubits[1])

    degree_sum = sum([graph.degree(n) for n in graph.nodes])

    return degree_sum / (num_qubits * (num_qubits - 1))


def compute_liveness(circuit: cirq.Circuit) -> float:
    """Compute the *liveness* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the liveness feature for this circuit.
    """
    return supermarq.converters.compute_liveness_with_qiskit(
        (supermarq.converters.cirq_to_qiskit(circuit))
    )


def compute_parallelism(circuit: cirq.Circuit) -> float:
    """Compute the parallelism of the given quantum circuit.

    Parallelism feature = max(1 - depth / # of gates, 0).

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the parallelism feature for this circuit.
    """
    depth = len(cirq.Circuit(circuit.all_operations()))
    return max(1 - (depth / len(list(circuit.all_operations()))), 0)


def compute_measurement(circuit: cirq.Circuit) -> float:
    """Compute the *measurement* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the measurement feature for this circuit.
    """
    return supermarq.converters.compute_measurement_with_qiskit(
        supermarq.converters.cirq_to_qiskit(circuit)
    )


def compute_entanglement(circuit: cirq.Circuit) -> float:
    """Compute the entanglement-ratio of the given quantum circuit.

    Entanglement-ratio = ratio between # of 2-qubit gates and total number of gates in the
    circuit.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the entanglement feature for this circuit.
    """
    return len(
        list(
            circuit.findall_operations(
                lambda op: isinstance(op, cirq.GateOperation) and len(op.qubits) == 2
            )
        )
    ) / len(list(circuit.all_operations()))


def compute_depth(circuit: cirq.Circuit) -> float:
    """Compute the *depth* feature of the input circuit.

    This function acts a wrapper which first converts the input `cirq.Circuit`
    into a `qiskit.QuantumCircuit` before calculating the feature value.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the depth feature for this circuit.
    """
    return supermarq.converters.compute_depth_with_qiskit(
        supermarq.converters.cirq_to_qiskit(circuit)
    )
