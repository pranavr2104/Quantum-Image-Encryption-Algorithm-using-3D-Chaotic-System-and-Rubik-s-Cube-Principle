from PIL import Image
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
import matplotlib.pyplot as plt


image_path = "lena.jpg"  
image = Image.open(image_path).convert("L")  
image = image.resize((4, 4))  
image_array = np.array(image)
image_array = image_array.astype(int)  


xor_mask = 0b10101010  
xor_image_array = np.bitwise_xor(image_array, xor_mask)


pos_qubits = int(np.log2(image_array.shape[0]) + np.log2(image_array.shape[1]))
intensity_qubits = 8
total_qubits = pos_qubits + intensity_qubits
qc = QuantumCircuit(total_qubits)

def position_to_binary(row, col, size):
    row_binary = format(row, f"0{int(np.log2(size[0]))}b")
    col_binary = format(col, f"0{int(np.log2(size[1]))}b")
    return row_binary + col_binary

def intensity_to_binary(value, depth):
    return format(value, f"0{depth}b")

for row in range(image_array.shape[0]):
    for col in range(image_array.shape[1]):
        pos_binary = position_to_binary(row, col, image_array.shape)
        intensity_binary = intensity_to_binary(xor_image_array[row, col], intensity_qubits)

        for idx, bit in enumerate(pos_binary):
            if bit == "1":
                qc.x(idx)

        for idx, bit in enumerate(intensity_binary):
            if bit == "1":
                qc.x(pos_qubits + idx)

        control_qubits = list(range(pos_qubits))
        target_qubits = list(range(pos_qubits, pos_qubits + intensity_qubits))
        qc.mcx(control_qubits, target_qubits[0])

        for idx, bit in enumerate(pos_binary):
            if bit == "1":
                qc.x(idx)


qc.measure_all()


fig, ax = plt.subplots(figsize=(15, 10))  
qc.draw(output='mpl', ax=ax)  
plt.title("NEQR Encoding Quantum Circuit", fontsize=16, pad=20)
plt.show()


simulator = AerSimulator()
transpiled_qc = transpile(qc, simulator)
result = simulator.run(transpiled_qc, shots=1024).result()
counts = result.get_counts()

print("Measurement Results:")
print(counts)
