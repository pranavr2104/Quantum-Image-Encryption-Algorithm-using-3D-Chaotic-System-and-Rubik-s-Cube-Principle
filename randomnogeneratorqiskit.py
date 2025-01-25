from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import requests
def ran_num(n):
    qc = QuantumCircuit(n,n)
    qc.h(range(n))
    qc.measure(range(n),range(n))
    sim = AerSimulator()
    comp = transpile(qc,sim)
    result = sim.run(comp,shots = 1).result()
    counts = result.get_counts()
    measured_bits = list(counts.keys())[0]

    rand_num= int(measured_bits, 2)
    return rand_num




def upload_to_thingspeak(rand_num, write_api_key):
    url = "https://api.thingspeak.com/update"
    params = {
        'api_key': write_api_key,  # Your ThingSpeak Write API Key
        'field1': rand_num        # The random number to upload
    }
    response = requests.post(url, params=params)
    if response.status_code == 200:
        print("Random number successfully uploaded to ThingSpeak!")
    else:
        print(f"Failed to upload to ThingSpeak. HTTP {response.status_code}: {response.text}")

n= 8
rand_num= ran_num(n)
print(f"Random {n}-bit number: {rand_num}")
write_api_key = "PIP0FHZP0WLGV8C1"
# Upload the random number to ThingSpeak
upload_to_thingspeak(rand_num, write_api_key)
