from .system import System
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs='?', default="c60.xyz")
    parser.add_argument("--cpu", action="store_true", help="Force CPU backend")
    args = parser.parse_args()

    sys = System(target_filename=args.target, n_iter=10, n_probe=10000, use_cpu=args.cpu)
    sys.run()
