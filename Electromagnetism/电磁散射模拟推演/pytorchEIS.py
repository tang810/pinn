import sys

from main import main


if __name__ == "__main__":
    # Keep legacy behavior: inverse training.
    sys.argv = [sys.argv[0], "--mode", "train", "--task", "inverse"] + sys.argv[1:]
    main()
