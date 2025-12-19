from massccs.system import System
import sys

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input.json")
        sys.exit(1)

    input_file = sys.argv[1]
    system = System(input_file)

if __name__ == "__main__":
    main()
