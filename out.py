import time

from tqdm import tqdm


def main():
    for number in tqdm(range(1, 101), desc='demo', position=0, dynamic_ncols=True):
        time.sleep(0.05)


if __name__ == '__main__':
    main()