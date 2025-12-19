import sys
import argparse
from massccs.system import System

def main():
    parser = argparse.ArgumentParser(description='MassCCS Python Implementation')
    parser.add_argument('input_file', help='Path to the input JSON file')
    args = parser.parse_args()

    try:
        system = System(args.input_file)
        system.run_simulation()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
