import cirq
import numpy as np
import qiskit


def cirq_to_qiskit(circuit: cirq.Circuit) -> qiskit.circuit.QuantumCircuit:
    """Convert a circuit from cirq to qiskit.

    Args:
        circuit: A `cirq.Circuit` to be converted.

    Returns:
        An equivalent `qiskit.QuantumCircuit`.
    """
    qasm = cirq.circuits.QasmOutput(circuit, tuple(sorted(circuit.all_qubits())))
    return qiskit.circuit.QuantumCircuit().from_qasm_str(str(qasm))


def compute_liveness_with_qiskit(circuit: qiskit.circuit.QuantumCircuit) -> float:
    """Compute the liveness of the given quantum circuit.

    Liveness feature = sum of all entries in the liveness matrix / (num_qubits * depth).

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the liveness feature for this circuit.
    """

    num_qubits = circuit.num_qubits
    dag = qiskit.converters.circuit_to_dag(circuit)
    dag.remove_all_ops_named("barrier")

    activity_matrix = np.zeros((num_qubits, dag.depth()))

    for i, layer in enumerate(dag.layers()):
        for op in layer["partition"]:
            for qubit in op:
                activity_matrix[circuit.find_bit(qubit).index, i] = 1

    return np.sum(activity_matrix) / (num_qubits * dag.depth())


def compute_measurement_with_qiskit(circuit: qiskit.circuit.QuantumCircuit) -> float:
    """Compute the measurement feature of the given quantum circuit.

    Measurement feature = # of layers of mid-circuit measurement / circuit depth.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the measurement feature for this circuit.
    """
    circuit.remove_final_measurements()
    dag = qiskit.converters.circuit_to_dag(circuit)
    dag.remove_all_ops_named("barrier")

    reset_moments = 0
    gate_depth = dag.depth()

    for layer in dag.layers():
        reset_present = False
        for op in layer["graph"].op_nodes():
            if op.name == "reset":
                reset_present = True
        if reset_present:
            reset_moments += 1

    return reset_moments / gate_depth


def compute_depth_with_qiskit(circuit: qiskit.circuit.QuantumCircuit) -> float:
    """Compute the critical depth of the given quantum circuit.

    Critical depth = # of 2-qubit gates along the critical path / total # of 2-qubit gates.

    Args:
        circuit: A quantum circuit.

    Returns:
        The value of the depth feature for this circuit.
    """
    dag = qiskit.converters.circuit_to_dag(circuit)
    dag.remove_all_ops_named("barrier")
    n_ed = 0
    two_q_gates = set([op.name for op in dag.two_qubit_ops()])
    for name in two_q_gates:
        try:
            n_ed += dag.count_ops_longest_path()[name]
        except KeyError:
            continue
    n_e = len(dag.two_qubit_ops())

    if n_ed == 0:
        return 0

    return n_ed / n_e
