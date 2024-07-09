def Clean():
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    size = 29
    for i in range(size-1, -1, -1): # i: 2 1 0
        print('-'*i*2, end='')
        j = size - i # j: 1 2 3
        for k in range(j):
            print(alphabet[size%len(alphabet) - k - 1], end='')
            if k < j-1:
                print('-', end='')
        for k in range(j-2, -1, -1):
            print('-', end='')
            print(alphabet[size%len(alphabet) - k - 1], end='')
        print('-'*i*2, end='')
        print()
    for i in range(1, size): # i: 2 1 0
        print('-'*i*2, end='')
        j = size - i # j: 1 2 3
        for k in range(j):
            print(alphabet[size%len(alphabet) - k - 1], end='')
            if k < j-1:
                print('-', end='')
        for k in range(j-2, -1, -1):
            print('-', end='')
            print(alphabet[size%len(alphabet) - k - 1], end='')
        print('-'*i*2, end='')
        print()
        
        
def Dirty(num=4):
    A = 'abcdefghijklmnopqrstuvwxyz'; size = num
    for i in range(size-1, -1, -1):
        print('-'*i*2, end=''); j = size - i 
        for k in range(j): print(A[size%len(A) - k - 1], end=''); print('-', end='') if k < j-1 else None
        for k in range(j-2, -1, -1): print('-', end=''); print(A[size%len(A) - k - 1], end='')
        print('-'*i*2)
    for i in range(1, size):
        print('-'*i*2, end=''); j = size - i 
        for k in range(j): print(A[size%len(A) - k - 1], end=''); print('-', end='') if k < j-1 else None
        for k in range(j-2, -1, -1): print('-', end=''); print(A[size%len(A) - k - 1], end='')
        print('-'*i*2)


if __name__ == '__main__':
    Dirty(40)