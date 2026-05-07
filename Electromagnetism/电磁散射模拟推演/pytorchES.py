import sys

from main import main


if __name__ == "__main__":
    # Keep legacy behavior: forward training.
    sys.argv = [sys.argv[0], "--mode", "train", "--task", "forward"] + sys.argv[1:]
    main()
