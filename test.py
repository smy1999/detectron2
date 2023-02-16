import argparse

import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument("--a", default="1")
    parser.add_argument("--b", default="2")
    parser.add_argument("--c", default="20")
    return parser


def run(a=7, b=9):
    print(a)
    print(b)


def test_parser():
    args = get_parser().parse_args()
    # print(args)
    run(**vars(args))


class a:
    def __init__(self):
        self.i = 1

    def __iter__(self):
        self.i = 99
        return self

    def __next__(self):
        self.i += 1
        return self.i


class b:
    def __init__(self):
        self.i = 1

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        return self.i + 1, self.i + 2, self.i + 3


def test_tqdm():
    oa = a()
    ob = b()
    for q, (w, e, r) in tqdm.tqdm(zip(oa, ob), total=3):
        print(q, w, e, r)
        if q > 10:
            break


if __name__ == "__main__":
    test_tqdm()

