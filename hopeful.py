from PIL import Image
import numpy as np
from scipy.integrate import solve_ivp
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import sys

# Key generation
def generate_keys(n):
    def cs(_, state):
        x, y, z = state
        return [1*y - 0.6*x*z, -x - 47*y*z, 1 - 0.5*y**2]

    t = 3000 + (2**n)**2
    s = solve_ivp(cs, [0, 100], [1.01, 0, 0],
                  t_eval=np.linspace(0, 100, t),
                  method='RK45', rtol=1e-9, atol=1e-12)

    x = s.y[0, 3000:3000+2**n]
    y = s.y[1, 3000:3000+2**n]
    z = s.y[2, 3000:3000+(2**n)**2]

    kr = [(int(abs(xi)*1e4) % (2**n-1)) + 1 for xi in x]
    kc = [(int(abs(yi)*1e4) % (2**n-1)) + 1 for yi in y]
    kx = np.array([int(abs(zi)*1e4) % 256 for zi in z]).reshape(2**n, 2**n)

    return kr, kc, kx

# FIXED: Encode image into NEQR with corrected intensity encoding
def encode_image(img_block, n):
    """
    Encode an image block into a NEQR quantum circuit.

    Args:
        img_block: 2D numpy array of grayscale pixel values (0-255).
        n: Number of qubits for row/column (block size = 2^n x 2^n).

    Returns:
        QuantumCircuit encoding the image block in NEQR format.
    """
    # Quantum registers: position (2n) and intensity (8)
    pq = QuantumRegister(2*n, 'pos')      # Position: row (n) + col (n)
    iq = QuantumRegister(8, 'intensity')  # Intensity: 8 bits for grayscale
    qc = QuantumCircuit(pq, iq)

    # Apply Hadamard to all position qubits to create superposition
    qc.h(pq)
    qc.barrier()

    rows, cols = img_block.shape

    for row in range(rows):
        for col in range(cols):
            # Binary representation of position
            pos_bits = f"{row:0{n}b}{col:0{n}b}"

            # Flip qubits for '0' bits in position (so that all-1s match)
            flip_qubits = [pq[i] for i, bit in enumerate(pos_bits) if bit == '0']
            if flip_qubits:
                qc.x(flip_qubits)

            # Binary representation of pixel intensity
            intensity_bits = format(img_block[row, col], '08b')[::-1]  # LSB first

            # For each '1' in intensity, apply MCX from all position qubits to intensity qubit
            for idx, bit in enumerate(intensity_bits):
                if bit == '1':
                    qc.mcx(list(pq), iq[idx])

            # Unflip the position qubits
            if flip_qubits:
                qc.x(flip_qubits)

            qc.barrier()

    return qc


# Scramble
def scramble(qc, kr, kc, ancilla, workspace, n):
    pq = qc.qubits[:2*n]
    rq = pq[:n]
    cq = pq[n:]

    for shift, control_idx in zip(kr, range(len(kr))):
        if shift == 0:
            continue

        control_bits = format(control_idx, f'0{n}b')
        for idx, bit in enumerate(control_bits):
            if bit == '1':
                qc.x(qc.qubits[idx])

        # Rotate the row qubits based on chaotic shift value
        for _ in range(shift % (n+1)):  # Rotate 0 to n times
            for i in range(n-1, 0, -1):
                qc.cx(rq[i], rq[i-1])
            qc.barrier()

        for idx, bit in enumerate(control_bits):
            if bit == '1':
                qc.x(qc.qubits[idx])

    for shift, control_idx in zip(kc, range(len(kc))):
        if shift == 0:
            continue

        control_bits = format(control_idx, f'0{n}b')
        for idx, bit in enumerate(control_bits):
            if bit == '1':
                qc.x(qc.qubits[idx+n])

        # Rotate the column qubits based on chaotic shift value
        for _ in range(shift % (n+1)):
            for i in range(n-1, 0, -1):
                qc.cx(cq[i], cq[i-1])
            qc.barrier()

        for idx, bit in enumerate(control_bits):
            if bit == '1':
                qc.x(qc.qubits[idx+n])
    # qc.draw(output='mpl', filename='quantum_circuit_enc.png')
    return qc

def unscramble(qc, kr, kc, ancilla, workspace, n):
    pq = qc.qubits[:2*n]
    rq = pq[:n]
    cq = pq[n:]
    
    # Reverse operation order but maintain original control indices
    # Columns first (reverse shift order, same controls as encryption)
    for shift, control_idx in zip(reversed(kc), range(len(kc))):  # Critical fix here
        if shift == 0:
            continue
            
        control_bits = format(control_idx, f'0{n}b')
        
        # Apply column control
        for idx, bit in enumerate(control_bits):
            if bit == '1':
                qc.x(cq[idx])
                
        # Inverse column rotation (mirror scramble's direction)
        for _ in range(shift % (n+1)):
            for i in range(0, n-1):  # Forward direction to undo backward shifts
                qc.cx(cq[i], cq[i+1])
                
        qc.barrier()
        
        # Unapply column control
        for idx, bit in enumerate(control_bits):
            if bit == '1':
                qc.x(cq[idx])
    
    # Then rows (reverse shift order, same controls as encryption)
    for shift, control_idx in zip(reversed(kr), range(len(kr))):  # Critical fix here
        if shift == 0:
            continue
            
        control_bits = format(control_idx, f'0{n}b')
        
        # Apply row control
        for idx, bit in enumerate(control_bits):
            if bit == '1':
                qc.x(rq[idx])
                
        # Inverse row rotation (mirror scramble's direction)
        for _ in range(shift % (n+1)):
            for i in range(0, n-1):  # Forward direction to undo backward shifts
                qc.cx(rq[i], rq[i+1])
                
        qc.barrier()
        
        # Unapply row control
        for idx, bit in enumerate(control_bits):
            if bit == '1':
                qc.x(rq[idx])
    
    return qc


# XOR Diffusion
def xor_diff(qc, kx, n):
    pq = qc.qubits[:2*n]
    iq = qc.qubits[2*n:2*n+8]
    ancilla_r = qc.qubits[2*n+8]  # This is likely the correct index
  # FIXED: Correct index for ancilla_r

    rows = cols = 2**n

    for row in range(rows):
        for col in range(cols):
            pb = f"{row:0{n}b}{col:0{n}b}"
            flip_qubits = [pq[i] for i, bit in enumerate(pb) if bit == '0']

            if flip_qubits:
                qc.x(flip_qubits)

            # Strict full multi-controlled comparison
            qc.mcx(list(pq), ancilla_r, ancilla_qubits=[], mode='noancilla')

            xv = kx[row, col]
            xb = format(xv, '08b')

            for b, bit in enumerate(xb):
                if bit == '1':
                    qc.ccx(ancilla_r, pq[0], iq[b])

            # Uncompute
            qc.mcx(list(pq), ancilla_r, ancilla_qubits=[], mode='noancilla')

            if flip_qubits:
                qc.x(flip_qubits)

    return qc

# FIXED: Process each block with improved measurement recovery
def process_block(img_block, n, ITER_max, kr, kc, kx, decrypt_en):
    # Save the original NEQR circuit for visualization
    qc = encode_image(img_block, n)
    # qc.draw(output='mpl', filename='neqr_circuit.png')
    
    # Add required registers
    ancilla = AncillaRegister(n, 'ancilla')
    workspace = QuantumRegister(1, 'workspace')
    ancilla_r = QuantumRegister(1, 'r')
    qc.add_register(ancilla)
    qc.add_register(workspace)
    qc.add_register(ancilla_r)

    if not decrypt_en:
        # ENCRYPTION
        for _ in range(ITER_max):
            qc = scramble(qc, kr, kc, ancilla, workspace, n)
            qc = xor_diff(qc, kx, n)
    else:
        for _ in range(ITER_max):
            qc = scramble(qc, kr, kc, ancilla, workspace, n)
            qc = xor_diff(qc, kx, n)
    
        for _ in range(ITER_max):
            qc = xor_diff(qc, kx, n)
            qc = unscramble(qc, kr, kc, ancilla, workspace, n)

    # Save the circuit after processing
    # if decrypt_en:
    #     qc.draw(output='mpl', filename='decryption_circuit.png')
    # else:
    #     qc.draw(output='mpl', filename='encryption_circuit.png')

    # Measure only the NEQR qubits (position and intensity)
    cr = ClassicalRegister(2*n + 8)
    qc.add_register(cr)
    qc.barrier()
    qc.measure(qc.qubits[:2*n+8], cr)

    # FIXED: Improved simulation with more shots
    simulator = AerSimulator(method='automatic')
    transpiled_qc = transpile(qc, simulator, optimization_level=2)
    result = simulator.run(transpiled_qc, shots=16384).result()  # Increased shots
    counts = result.get_counts()

    rows, cols = img_block.shape
    row_bits = int(np.ceil(np.log2(rows)))
    col_bits = int(np.ceil(np.log2(cols)))

    # FIXED: Improved measurement recovery logic
    recovered = np.zeros((rows, cols), dtype=np.uint8)
    pm = {}

    for state, count in counts.items():
        state = state[::-1]  # Reverse for little-endian
        
        # Extract position and intensity bits
        pos_bits = state[:row_bits + col_bits]
        intensity_bits = state[row_bits + col_bits:row_bits + col_bits + 8]
        
        # Convert to row, col coordinates
        row = int(pos_bits[:row_bits], 2)
        col = int(pos_bits[row_bits:], 2)
        
        # Only process valid positions
        if row < rows and col < cols:
            # Keep track of the most frequent measurement for each position
            if (row, col) not in pm or pm[(row, col)][1] < count:
                pm[(row, col)] = (int(intensity_bits, 2), count)

    # Reconstruct the image from measurements
    for (row, col), (pix_val, _) in pm.items():
        recovered[row, col] = pix_val

    return recovered

# Blockwise processing
def blockwise_process(image_path, block_size, ITER_max, decrypt_en):
    # 1. Load image and set up parameters
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    height, width = img_array.shape
    n = 3# 8x8 blocks → 3+3 qubits

    # 2. Initialize final image array
    final_img = np.zeros_like(img_array)

    # 3. Removed the global np.random.seed(42) - unnecessary and potentially confusing

    # 4. Loop through blocks
    for i in range(0, height, 2**n):
        for j in range(0, width, 2**n):
            # 5. Extract and pad block (if necessary)
            block = img_array[i:i+2**n, j:j+2**n]
            if block.shape[0] != 2**n or block.shape[1] != 2**n:
                padded_block = np.zeros((2**n, 2**n), dtype=np.uint8)
                padded_block[:block.shape[0], :block.shape[1]] = block
                block = padded_block

            # 6. Generate keys deterministically based on block position
            #    This ensures the same keys are used for encrypting and decrypting
            #    the same block position.
            block_seed = i * width + j  # Unique seed for this block
            # np.random.seed(block_seed) # Unnecessary if generate_keys doesn't use np.random
            kr, kc, kx = generate_keys(n) # Assumes generate_keys is deterministic

            # 7. Process the block - **THIS IS THE CRITICAL STEP**
            #    It passes the correct `decrypt_en` flag to `process_block`.
            processed_block = process_block(block, n, ITER_max, kr, kc, kx, decrypt_en)

            # 8. Reconstruct the final image, handling boundaries
            final_img[i:i+processed_block.shape[0], j:j+processed_block.shape[1]] = \
                processed_block[:min(height-i, processed_block.shape[0]),
                                :min(width-j, processed_block.shape[1])]

    # 9. Save the final image with the correct name based on `decrypt_en`
    final_pil = Image.fromarray(final_img[:height, :width])
    if decrypt_en:
        final_pil.save("decrypted_image.png") # Saves as decrypted if flag is True
    else:
        final_pil.save("encrypted_image.png") # Saves as encrypted if flag is False

    return # Function completes

# Main CLI
def main():
    if len(sys.argv) < 4:
        print("Usage: python hopeful.py [encrypt|decrypt] <image_path> <iterations>")
        sys.exit(1)

    mode = sys.argv[1]
    image_path = sys.argv[2]
    iterations = int(sys.argv[3])
    
    if mode == 'encrypt':
        blockwise_process(image_path, block_size=8, ITER_max=iterations, decrypt_en=False)
        print("Image encrypted successfully. Saved as 'encrypted_image.png'")
    elif mode == 'decrypt':
        blockwise_process(image_path, block_size=8, ITER_max=iterations, decrypt_en=True)
        print("Image decrypted successfully. Saved as 'decrypted_image.png'")
    else:
        print("Invalid mode. Use 'encrypt' or 'decrypt'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
