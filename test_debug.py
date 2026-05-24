import sys
import os

print("\n===== SYSTEM PATHS =====")
for path in sys.path:
    print(path)

print("\n===== TESTING IMPORT =====")
try:
    import langchain
    print("Langchain path:", langchain.__file__)
    
    # ఇక్కడ ఎక్కడ ఫెయిల్ అవుతుందో చూద్దాం
    from langchain import chains
    print("Chains imported successfully!")
except Exception as e:
    print("Error occurred:", str(e))