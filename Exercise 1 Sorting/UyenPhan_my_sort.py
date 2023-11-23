def main():
    list_num = input("Give a list of integers separated by space: ")
    list_num = list_num.split()
    list_num = [int(x) for x in list_num]
    list_num.sort()
    print(f"Given numbers sorted: {list_num}")


if __name__ == "__main__":
    main()
