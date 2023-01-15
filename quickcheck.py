import pickle
import os
from learner import *
from learner_settings import *

print(f"TARGET DIRECTORY: {TARGET_DIR}")
print(f"TARGET ARCHITECTURE: {TARGET_ARCHITECTURE}")
gen_start = 0
while os.path.exists(f"{TARGET_DIR}/{TARGET_ARCHITECTURE}-gen-{gen_start}.pickle"):
    gen_start += 1

if gen_start == 0:
    print("Nothing found")
else:
    with open(f"{TARGET_DIR}/{TARGET_ARCHITECTURE}-gen-{gen_start - 1}.pickle", 'rb') as f:
        genobj = pickle.load(f)
    print(genobj.population[0].network.dump())
    input()
    for gen_start in range(1, gen_start + 1):
        with open(f"{TARGET_DIR}/{TARGET_ARCHITECTURE}-gen-{gen_start - 1}.pickle", 'rb') as f:
            genobj = pickle.load(f)
        assert isinstance(genobj, Generation)
        print(genobj)

