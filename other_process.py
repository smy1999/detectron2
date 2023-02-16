def run():
    while True:
        line = input()
        if line == '.':
            break
        key = line.split('=')[0].strip()
        line = key + ' = args.' + key
        print(line)
        # print('print("' + key + '", ' + key + ')')


if __name__ == "__main__":
    run()
